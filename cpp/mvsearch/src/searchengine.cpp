// движок умного поиска на c++: словари, регулярки, марковский типизатор, RecSys
#include "searchengine.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonDocument>
#include <QSet>
#include <QTextStream>
#include <algorithm>
#include <cmath>

namespace {

// алиасы брендов: пользователь пишет «айфон», имеет в виду Apple
const QHash<QString, QString> kBrandAliases = {
    {u"iphone"_qs, u"Apple"_qs},   {u"айфон"_qs, u"Apple"_qs},   {u"айфоны"_qs, u"Apple"_qs},
    {u"macbook"_qs, u"Apple"_qs},  {u"макбук"_qs, u"Apple"_qs},  {u"ipad"_qs, u"Apple"_qs},
    {u"айпад"_qs, u"Apple"_qs},    {u"airpods"_qs, u"Apple"_qs}, {u"аирподс"_qs, u"Apple"_qs},
    {u"galaxy"_qs, u"Samsung"_qs}, {u"самсунг"_qs, u"Samsung"_qs},
    {u"редми"_qs, u"Xiaomi"_qs},   {u"redmi"_qs, u"Xiaomi"_qs},  {u"poco"_qs, u"Xiaomi"_qs},
    {u"поко"_qs, u"Xiaomi"_qs},    {u"сяоми"_qs, u"Xiaomi"_qs},  {u"ксяоми"_qs, u"Xiaomi"_qs},
    {u"honor"_qs, u"HONOR"_qs},    {u"хонор"_qs, u"HONOR"_qs},
    {u"хуавей"_qs, u"HUAWEI"_qs},  {u"леново"_qs, u"Lenovo"_qs}, {u"ленова"_qs, u"Lenovo"_qs},
    {u"асус"_qs, u"ASUS"_qs},      {u"асер"_qs, u"Acer"_qs},     {u"тошиба"_qs, u"Toshiba"_qs},
    {u"хайер"_qs, u"Haier"_qs},    {u"китфорт"_qs, u"Kitfort"_qs},
    {u"плейстейшен"_qs, u"Sony"_qs}, {u"playstation"_qs, u"Sony"_qs}, {u"пс5"_qs, u"Sony"_qs},
    {u"ps5"_qs, u"Sony"_qs},       {u"ps4"_qs, u"Sony"_qs},      {u"дайсон"_qs, u"Dyson"_qs},
};

const QSet<QString> kColors = {
    u"белый"_qs,  u"черный"_qs, u"чёрный"_qs, u"серый"_qs,   u"красный"_qs, u"синий"_qs,
    u"зеленый"_qs, u"зелёный"_qs, u"золотой"_qs, u"розовый"_qs, u"голубой"_qs, u"фиолетовый"_qs,
    u"оранжевый"_qs, u"коричневый"_qs, u"бежевый"_qs, u"желтый"_qs, u"жёлтый"_qs,
    u"титановый"_qs, u"графитовый"_qs, u"серебристый"_qs, u"бирюзовый"_qs, u"бордовый"_qs,
    u"black"_qs,  u"white"_qs,  u"silver"_qs, u"gold"_qs,    u"graphite"_qs, u"titanium"_qs,
    u"midnight"_qs, u"starlight"_qs, u"pink"_qs, u"blue"_qs,  u"green"_qs,
};

QString numPat()
{
    return QStringLiteral("(?:(?:от\\s*)?\\d+(?:[.,/]\\d+)?\\s*(?:-|до|—)\\s*)?\\d+(?:[.,/]\\d+)?");
}

} // namespace

SearchEngine::SearchEngine(QObject *parent)
    : QObject(parent)
{
    m_brandAliases = kBrandAliases;
    m_colors = kColors;

    const QString num = numPat();
    auto rx = [](const QString &p) {
        return QRegularExpression(p, QRegularExpression::CaseInsensitiveOption
                                         | QRegularExpression::UseUnicodePropertiesOption);
    };
    m_attrPatterns = {
        {rx(QStringLiteral("\\b(%1)\\s*(гб|gb|тб|tb|гиг(?:ов)?|мб|mb)\\b").arg(num)), u"память"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(мм|mm|см|cm|дюйм(?:а|ов)?|\")\\b").arg(num)), u"размер"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(вт|w|ватт|квт|kw)\\b").arg(num)), u"мощность"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(кг|kg|грамм|г)\\b").arg(num)), u"вес"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(л|литр(?:а|ов)?|мл|ml)\\b").arg(num)), u"объём"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(гц|hz|кгц|мгц)\\b").arg(num)), u"частота"_qs},
        {rx(QStringLiteral("\\b(4k|4к|8k|8к|1080p|720p|1440p|uhd|full\\s*hd|fhd)\\b")), u"разрешение"_qs},
        {rx(QStringLiteral("\\b(wi[- ]?fi|bluetooth|nfc|5g|4g|lte|gps|usb[- ]?c|type[- ]?c|hdmi)\\b")), u"связь"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*(шт|штук[иа]?)\\b").arg(num)), u"количество"_qs},
        {rx(QStringLiteral("\\b(%1)\\s*[xх×*]\\s*(\\d+(?:[.,]\\d+)?)\\b").arg(num)), u"габариты"_qs},
    };

    loadData();
}

QString SearchEngine::dataDir() const
{
    // data рядом с exe (установленное приложение) либо на два уровня выше (dev-сборка)
    const QString appDir = QCoreApplication::applicationDirPath();
    for (const QString &candidate : {appDir + u"/data"_qs,
                                     appDir + u"/../data"_qs,
                                     appDir + u"/../../data"_qs,
                                     appDir + u"/../../../data"_qs}) {
        if (QDir(candidate).exists(u"brands.txt"_qs))
            return QDir(candidate).absolutePath();
    }
    return appDir + u"/data"_qs;
}

void SearchEngine::loadData()
{
    const QString dir = dataDir();

    auto readLines = [](const QString &path) {
        QStringList out;
        QFile f(path);
        if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream st(&f);
            st.setEncoding(QStringConverter::Utf8);
            while (!st.atEnd()) {
                const QString line = st.readLine().trimmed();
                if (!line.isEmpty())
                    out << line;
            }
        }
        return out;
    };

    for (const QString &b : readLines(dir + u"/brands.txt"_qs))
        m_brandCanonical.insert(normalize(b), b);
    for (const QString &c : readLines(dir + u"/categories.txt"_qs))
        m_categories.insert(normalize(c));
    for (const QString &m : readLines(dir + u"/model_phrases.txt"_qs))
        m_modelPhrases.insert(normalize(m));

    // марковский типизатор: берём argmax по каждой таблице
    QFile mf(dir + u"/markov_typer.json"_qs);
    if (mf.open(QIODevice::ReadOnly)) {
        const QJsonObject root = QJsonDocument::fromJson(mf.readAll()).object();
        auto fillArgmax = [](const QJsonObject &table, QHash<QString, QString> &out) {
            for (auto it = table.begin(); it != table.end(); ++it) {
                const QJsonObject counts = it.value().toObject();
                QString best;
                double bestN = -1;
                for (auto c = counts.begin(); c != counts.end(); ++c) {
                    if (c.value().toDouble() > bestN) {
                        bestN = c.value().toDouble();
                        best = c.key();
                    }
                }
                if (!best.isEmpty())
                    out.insert(it.key(), best);
            }
        };
        fillArgmax(root.value(u"bigram_to_type"_qs).toObject(), m_bigramType);
        fillArgmax(root.value(u"unit_to_type"_qs).toObject(), m_unitType);
    }

    // каталог для RecSys
    QFile cf(dir + u"/catalog.json"_qs);
    if (cf.open(QIODevice::ReadOnly)) {
        const QJsonArray arr = QJsonDocument::fromJson(cf.readAll()).array();
        m_catalog.reserve(arr.size());
        for (const QJsonValue &v : arr) {
            const QJsonObject o = v.toObject();
            CatalogItem item;
            item.skuId = static_cast<qint64>(o.value(u"sku_id"_qs).toDouble());
            item.name = o.value(u"name"_qs).toString();
            item.nameLower = normalize(item.name);
            item.brand = o.value(u"brand"_qs).toString();
            item.brandLower = normalize(item.brand);
            item.price = o.value(u"price"_qs).toDouble();
            item.tokens = tokenize(item.nameLower);
            m_catalog.push_back(item);
        }
    }

    m_ready = !m_brandCanonical.isEmpty() && !m_catalog.isEmpty();
    emit readyChanged();
}

QString SearchEngine::normalize(const QString &text)
{
    QString t = text.toLower();
    t.replace(u'ё', u'е');
    return t.simplified();
}

QString SearchEngine::splitGlued(const QString &text)
{
    // «16гб» -> «16 гб», «128gb» -> «128 gb»
    static const QRegularExpression re(QStringLiteral("(\\d+)([А-Яа-яA-Za-z]{1,4})\\b"),
                                       QRegularExpression::UseUnicodePropertiesOption);
    QString out = text;
    out.replace(re, QStringLiteral("\\1 \\2"));
    return out;
}

QStringList SearchEngine::tokenize(const QString &text) const
{
    static const QRegularExpression re(QStringLiteral("[^a-zа-я0-9ё.+]+"),
                                       QRegularExpression::CaseInsensitiveOption
                                           | QRegularExpression::UseUnicodePropertiesOption);
    QStringList toks = text.split(re, Qt::SkipEmptyParts);
    return toks;
}

QString SearchEngine::markovType(const QStringList &spanTokens) const
{
    // нормализуем как в python: числа -> <num>
    static const QRegularExpression numRe(QStringLiteral("^\\d+(?:[.,/]\\d+)?$"));
    QStringList norm;
    for (const QString &t : spanTokens)
        norm << (numRe.match(t).hasMatch() ? u"<num>"_qs : t);

    for (int i = 0; i + 1 < norm.size(); ++i) {
        const QString key = norm[i] + u'|' + norm[i + 1];
        if (m_bigramType.contains(key))
            return m_bigramType.value(key);
    }
    for (const QString &t : norm) {
        if (t != u"<num>"_qs && m_unitType.contains(t))
            return m_unitType.value(t);
    }
    return QString();
}

QVariantMap SearchEngine::extract(const QString &query)
{
    QElapsedTimer timer;
    timer.start();

    QVariantMap result;
    result[u"query"_qs] = query;

    const QString prepared = splitGlued(normalize(query));
    const QStringList tokens = tokenize(prepared);

    QString brand, category, model;
    QVariantList entities;
    QVariantMap attributes;
    QSet<int> usedTokens;

    // --- 1. атрибуты по регуляркам + марковский тип
    for (const auto &pat : m_attrPatterns) {
        auto it = pat.first.globalMatch(prepared);
        while (it.hasNext()) {
            const auto m = it.next();
            const QString spanText = m.captured(0).trimmed();
            const QStringList spanTokens = tokenize(spanText);
            QString type = markovType(spanTokens);
            if (type.isEmpty())
                type = pat.second;
            attributes[type] = spanText;

            QVariantMap ent;
            ent[u"text"_qs] = spanText;
            ent[u"label"_qs] = u"ATTR"_qs;
            ent[u"type"_qs] = type;
            entities.append(ent);
            for (int i = 0; i < tokens.size(); ++i)
                if (spanTokens.contains(tokens[i]))
                    usedTokens.insert(i);
        }
    }

    // --- 2. цвет
    for (int i = 0; i < tokens.size(); ++i) {
        if (m_colors.contains(tokens[i])) {
            attributes[u"цвет"_qs] = tokens[i];
            QVariantMap ent;
            ent[u"text"_qs] = tokens[i];
            ent[u"label"_qs] = u"ATTR"_qs;
            ent[u"type"_qs] = u"цвет"_qs;
            entities.append(ent);
            usedTokens.insert(i);
        }
    }

    // --- 3. бренд: сначала точное слово/фраза, затем алиасы
    int brandTokenEnd = -1;
    for (int n = 3; n >= 1 && brand.isEmpty(); --n) {
        for (int i = 0; i + n <= tokens.size(); ++i) {
            QStringList slice = tokens.mid(i, n);
            const QString phrase = slice.join(u' ');
            if (m_brandCanonical.contains(phrase)) {
                brand = m_brandCanonical.value(phrase);
                for (int k = i; k < i + n; ++k) usedTokens.insert(k);
                brandTokenEnd = i + n;
                QVariantMap ent;
                ent[u"text"_qs] = phrase;
                ent[u"label"_qs] = u"BRAND"_qs;
                entities.append(ent);
                break;
            }
        }
    }
    if (brand.isEmpty()) {
        for (int i = 0; i < tokens.size(); ++i) {
            if (m_brandAliases.contains(tokens[i])) {
                brand = m_brandAliases.value(tokens[i]);
                usedTokens.insert(i);
                brandTokenEnd = i + 1;
                QVariantMap ent;
                ent[u"text"_qs] = tokens[i];
                ent[u"label"_qs] = u"BRAND"_qs;
                ent[u"note"_qs] = u"алиас"_qs;
                entities.append(ent);
                break;
            }
        }
    }

    // --- 4. категория (фразы до 3 токенов)
    for (int n = 3; n >= 1 && category.isEmpty(); --n) {
        for (int i = 0; i + n <= tokens.size(); ++i) {
            const QString phrase = tokens.mid(i, n).join(u' ');
            if (m_categories.contains(phrase)) {
                category = phrase;
                for (int k = i; k < i + n; ++k) usedTokens.insert(k);
                QVariantMap ent;
                ent[u"text"_qs] = phrase;
                ent[u"label"_qs] = u"CATEGORY"_qs;
                entities.append(ent);
                break;
            }
        }
    }

    // --- 5. модель: словарь линеек из майнинга; иначе хвост после бренда
    for (int n = 4; n >= 1 && model.isEmpty(); --n) {
        for (int i = 0; i + n <= tokens.size(); ++i) {
            const QString phrase = tokens.mid(i, n).join(u' ');
            if (m_modelPhrases.contains(phrase)) {
                model = phrase;
                for (int k = i; k < i + n; ++k) usedTokens.insert(k);
                QVariantMap ent;
                ent[u"text"_qs] = phrase;
                ent[u"label"_qs] = u"MODEL"_qs;
                ent[u"note"_qs] = u"словарь линеек"_qs;
                entities.append(ent);
                break;
            }
        }
    }
    if (model.isEmpty() && brandTokenEnd >= 0) {
        // хвост после бренда: свободные латинские/цифровые токены
        static const QRegularExpression latinRe(QStringLiteral("^[a-z0-9.+-]+$"));
        QStringList tail;
        for (int i = brandTokenEnd; i < tokens.size() && tail.size() < 4; ++i) {
            if (usedTokens.contains(i))
                break;
            if (!latinRe.match(tokens[i]).hasMatch())
                break;
            tail << tokens[i];
            usedTokens.insert(i);
        }
        if (!tail.isEmpty()) {
            model = tail.join(u' ');
            QVariantMap ent;
            ent[u"text"_qs] = model;
            ent[u"label"_qs] = u"MODEL"_qs;
            ent[u"note"_qs] = u"хвост после бренда"_qs;
            entities.append(ent);
        }
    }

    result[u"brand"_qs] = brand;
    result[u"category"_qs] = category;
    result[u"model"_qs] = model;
    result[u"attributes"_qs] = attributes;
    result[u"entities"_qs] = entities;
    result[u"tokens"_qs] = tokens;
    result[u"latency_ms"_qs] = timer.nsecsElapsed() / 1e6;
    return result;
}

QVariantList SearchEngine::rankCatalog(const QVariantMap &facts, int topN)
{
    // ранжирование ТОЛЬКО по фактам: бренд, категория, модель, атрибуты + косинус по токенам
    const QString brand = normalize(facts.value(u"brand"_qs).toString());
    const QString category = normalize(facts.value(u"category"_qs).toString());
    const QString model = normalize(facts.value(u"model"_qs).toString());
    const QVariantMap attributes = facts.value(u"attributes"_qs).toMap();

    QStringList factTokens;
    if (!category.isEmpty()) factTokens += tokenize(category);
    if (!model.isEmpty()) factTokens += tokenize(model);
    for (auto it = attributes.begin(); it != attributes.end(); ++it)
        factTokens += tokenize(normalize(it.value().toString()));

    struct Scored { double score; const CatalogItem *item; QString why; };
    QVector<Scored> scored;
    scored.reserve(m_catalog.size());

    for (const CatalogItem &item : m_catalog) {
        double score = 0;
        QStringList why;

        if (!brand.isEmpty()) {
            if (item.brandLower == brand || item.nameLower.contains(brand)) {
                score += 3.0;
                why << u"бренд"_qs;
            } else {
                score -= 1.2; // чужой бренд при заданном бренде — штраф
            }
        }
        if (!category.isEmpty() && item.nameLower.contains(category)) {
            score += 2.4;
            why << u"категория"_qs;
        }
        if (!model.isEmpty() && item.nameLower.contains(model)) {
            score += 2.8;
            why << u"модель"_qs;
        }

        int attrHits = 0;
        for (auto it = attributes.begin(); it != attributes.end(); ++it) {
            const QString val = normalize(it.value().toString());
            if (!val.isEmpty() && item.nameLower.contains(val))
                ++attrHits;
        }
        if (attrHits > 0) {
            score += 1.2 * attrHits;
            why << QStringLiteral("атрибуты ×%1").arg(attrHits);
        }

        // косинус по токенам фактов и названия
        if (!factTokens.isEmpty() && !item.tokens.isEmpty()) {
            int common = 0;
            for (const QString &t : factTokens)
                if (item.tokens.contains(t))
                    ++common;
            const double cos = common / (std::sqrt(double(factTokens.size()))
                                         * std::sqrt(double(item.tokens.size())));
            score += 1.5 * cos;
        }

        if (score > 0.15)
            scored.push_back({score, &item, why.join(u" · "_qs)});
    }

    std::partial_sort(scored.begin(),
                      scored.begin() + std::min<qsizetype>(topN, scored.size()),
                      scored.end(),
                      [](const Scored &a, const Scored &b) { return a.score > b.score; });

    QVariantList out;
    const int n = std::min<int>(topN, scored.size());
    const double maxScore = n > 0 ? scored[0].score : 1.0;
    for (int i = 0; i < n; ++i) {
        QVariantMap row;
        row[u"name"_qs] = scored[i].item->name;
        row[u"brand"_qs] = scored[i].item->brand;
        row[u"price"_qs] = scored[i].item->price;
        row[u"skuId"_qs] = scored[i].item->skuId;
        row[u"score"_qs] = scored[i].score;
        row[u"match"_qs] = maxScore > 0 ? scored[i].score / maxScore : 0;
        row[u"why"_qs] = scored[i].why;
        out.append(row);
    }
    return out;
}

QString SearchEngine::extractJson(const QString &query)
{
    const QVariantMap m = extract(query);
    return QString::fromUtf8(
        QJsonDocument(QJsonObject::fromVariantMap(m)).toJson(QJsonDocument::Indented));
}

QString SearchEngine::envQuery() const
{
    return qEnvironmentVariable("MV_QUERY");
}

QString SearchEngine::envShootPath() const
{
    return qEnvironmentVariable("MV_SHOOT");
}
