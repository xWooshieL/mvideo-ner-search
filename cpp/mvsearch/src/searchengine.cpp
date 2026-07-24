// движок умного поиска на c++: словари, регулярки, марковский типизатор, RecSys
#include "searchengine.h"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QGuiApplication>
#include <QJsonDocument>
#include <QProcess>
#include <QSet>
#include <QSettings>
#include <QTextStream>
#include <QTimer>
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
    // data рядом с exe (Windows) либо в Resources бандла (macOS .app), либо выше (dev-сборка)
    const QString appDir = QCoreApplication::applicationDirPath();
    for (const QString &candidate : {appDir + u"/data"_qs,
                                     appDir + u"/../Resources/data"_qs,
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

    for (const QString &b : readLines(dir + u"/brands.txt"_qs)) {
        const QString key = normalize(b);
        m_brandCanonical.insert(key, b);
        if (!key.contains(u' '))
            m_spellVocab.insert(key);
    }
    for (const QString &c : readLines(dir + u"/categories.txt"_qs)) {
        const QString key = normalize(c);
        m_categories.insert(key);
        if (!key.contains(u' '))
            m_spellVocab.insert(key);
    }
    for (const QString &m : readLines(dir + u"/model_phrases.txt"_qs))
        m_modelPhrases.insert(normalize(m));

    // единицы + алиасы SpellFix (spell_aliases.txt: canon\talias1,alias2)
    for (const QString &u : {u"гб"_qs, u"gb"_qs, u"тб"_qs, u"tb"_qs, u"мб"_qs, u"mb"_qs,
                             u"мм"_qs, u"см"_qs, u"вт"_qs, u"кг"_qs, u"л"_qs, u"мл"_qs})
        m_spellVocab.insert(u);
    for (const QString &line : readLines(dir + u"/spell_aliases.txt"_qs)) {
        if (line.startsWith(u'#'))
            continue;
        const QStringList parts = line.split(u'\t');
        if (parts.size() < 2)
            continue;
        const QString canon = normalize(parts[0]);
        for (const QString &aliasRaw : parts[1].split(u',')) {
            const QString alias = normalize(aliasRaw.trimmed());
            if (!alias.isEmpty() && alias != canon)
                m_spellAliases.insert(alias, canon);
        }
    }

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

bool SearchEngine::hasMixedScript(const QString &token)
{
    bool cyr = false, lat = false;
    for (const QChar &ch : token) {
        if (ch.isLetter()) {
            if (ch.script() == QChar::Script_Cyrillic)
                cyr = true;
            else if (ch.script() == QChar::Script_Latin)
                lat = true;
        }
    }
    return cyr && lat;
}

QString SearchEngine::normalizeHomoglyphs(const QString &token) const
{
    if (!hasMixedScript(token))
        return token;

    static const QHash<QChar, QChar> latToCyr = {
        {u'a', u'а'}, {u'c', u'с'}, {u'e', u'е'}, {u'o', u'о'},
        {u'p', u'р'}, {u'x', u'х'}, {u'y', u'у'},
        {u'A', u'А'}, {u'C', u'С'}, {u'E', u'Е'}, {u'O', u'О'},
        {u'P', u'Р'}, {u'X', u'Х'}, {u'Y', u'У'}, {u'B', u'В'},
        {u'H', u'Н'}, {u'K', u'К'}, {u'M', u'М'}, {u'T', u'Т'},
    };
    QHash<QChar, QChar> cyrToLat;
    for (auto it = latToCyr.begin(); it != latToCyr.end(); ++it)
        cyrToLat.insert(it.value(), it.key());

    int cyrOnly = 0, latOnly = 0;
    for (const QChar &ch : token) {
        if (!ch.isLetter())
            continue;
        const bool isHomo = latToCyr.contains(ch) || cyrToLat.contains(ch);
        if (isHomo)
            continue;
        if (ch.script() == QChar::Script_Cyrillic)
            ++cyrOnly;
        else if (ch.script() == QChar::Script_Latin)
            ++latOnly;
    }
    if (cyrOnly == 0 && latOnly == 0)
        return token;

    const bool toCyr = cyrOnly >= latOnly;
    QString out;
    out.reserve(token.size());
    for (const QChar &ch : token) {
        if (toCyr && latToCyr.contains(ch))
            out.append(latToCyr.value(ch));
        else if (!toCyr && cyrToLat.contains(ch))
            out.append(cyrToLat.value(ch));
        else
            out.append(ch);
    }
    return out;
}

int SearchEngine::editDistance(const QString &a, const QString &b)
{
    const int n = a.size(), m = b.size();
    if (qAbs(n - m) > 2)
        return 99;
    QVector<int> prev(m + 1), cur(m + 1);
    for (int j = 0; j <= m; ++j)
        prev[j] = j;
    for (int i = 1; i <= n; ++i) {
        cur[0] = i;
        for (int j = 1; j <= m; ++j) {
            const int cost = a[i - 1] == b[j - 1] ? 0 : 1;
            cur[j] = qMin(qMin(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
        }
        prev.swap(cur);
    }
    return prev[m];
}

QString SearchEngine::bestFuzzyCanon(const QString &token) const
{
    const QString t = normalize(token);
    if (t.size() < 4 || m_spellVocab.contains(t))
        return QString();
    QString best;
    int bestDist = 99;
    for (const QString &cand : m_spellVocab) {
        if (qAbs(cand.size() - t.size()) > 2)
            continue;
        const int d = editDistance(t, cand);
        if (d < bestDist && d <= qMax(1, t.size() / 3)) {
            bestDist = d;
            best = cand;
            if (d == 0)
                break;
        }
    }
    return bestDist <= 1 || (bestDist == 2 && t.size() >= 6) ? best : QString();
}

QString SearchEngine::fixTokenSpell(const QString &token) const
{
    const QString t0 = normalize(token);
    if (m_spellAliases.contains(t0))
        return m_spellAliases.value(t0);

    const QString homo = normalizeHomoglyphs(token);
    const QString homoN = normalize(homo);
    if (m_spellAliases.contains(homoN))
        return m_spellAliases.value(homoN);

    // 16гь / 16гю → 16 гб
    static const QRegularExpression glued(QStringLiteral("^(\\d+)([а-яёa-z]{1,5})$"),
                                          QRegularExpression::CaseInsensitiveOption
                                              | QRegularExpression::UseUnicodePropertiesOption);
    const auto gm = glued.match(homoN);
    if (gm.hasMatch()) {
        const QString num = gm.captured(1);
        const QString unit = gm.captured(2);
        static const QStringList units = {
            u"гб"_qs, u"gb"_qs, u"тб"_qs, u"tb"_qs, u"мб"_qs, u"mb"_qs,
            u"мм"_qs, u"см"_qs, u"вт"_qs, u"кг"_qs, u"л"_qs, u"мл"_qs
        };
        if (units.contains(unit))
            return num + u' ' + unit;
        QString bestUnit;
        int bestD = 99;
        for (const QString &u : units) {
            const int d = editDistance(unit, u);
            if (d < bestD) {
                bestD = d;
                bestUnit = u;
            }
        }
        if (bestD <= 1 && !bestUnit.isEmpty())
            return num + u' ' + bestUnit;
    }

    const QString fuzzy = bestFuzzyCanon(homoN.isEmpty() ? token : homo);
    if (!fuzzy.isEmpty())
        return fuzzy;
    return homoN != t0 ? homo : token;
}

QString SearchEngine::spellFixQuery(const QString &query) const
{
    if (query.isEmpty())
        return query;
    static const QRegularExpression wordRe(QStringLiteral("(\\w+|\\W+)"),
                                           QRegularExpression::UseUnicodePropertiesOption);
    QString out;
    auto it = wordRe.globalMatch(query);
    while (it.hasNext()) {
        const QString part = it.next().captured(1);
        if (part.isEmpty())
            continue;
        if (!part[0].isLetterOrNumber()) {
            out += part;
            continue;
        }
        if (part[0].isDigit() && part.mid(1).isEmpty()) {
            out += part;
            continue;
        }
        out += fixTokenSpell(part);
    }
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

    const QString fixed = spellFixQuery(query);
    if (fixed != query)
        result[u"query_fixed"_qs] = fixed;

    const QString prepared = splitGlued(normalize(fixed));
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

void SearchEngine::prepareForUninstall(bool deleteSettings)
{
    if (deleteSettings) {
        QSettings settings(QStringLiteral("MVideo"), QStringLiteral("MvSearch"));
        settings.clear();
        settings.sync();
    }

    /* строки "1"/"0" — деинсталлятор читает RegQueryStringValue, не DWORD */
    QSettings uninstallFlags(QStringLiteral("MVideo"), QStringLiteral("UninstallMvSearch"));
    uninstallFlags.setValue(
        QStringLiteral("DeleteSettings"),
        deleteSettings ? QStringLiteral("1") : QStringLiteral("0"));
    uninstallFlags.sync();
}

void SearchEngine::launchUninstaller()
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QFileInfoList candidates = QDir(appDir).entryInfoList(
        QStringList{QStringLiteral("unins*.exe")}, QDir::Files, QDir::Name);

    if (candidates.isEmpty())
        return;

    QFileInfo best = candidates.first();
    for (const QFileInfo &candidate : candidates) {
        if (candidate.fileName().compare(best.fileName(), Qt::CaseInsensitive) > 0)
            best = candidate;
    }

    if (!QProcess::startDetached(best.absoluteFilePath(), {QStringLiteral("/SILENT")}))
        return;

    QTimer::singleShot(200, qApp, &QCoreApplication::quit);
}
