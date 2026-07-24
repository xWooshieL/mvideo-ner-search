// движок умного поиска: словари + регулярки + марковский типизатор + RecSys по фактам
#pragma once

#include <QHash>
#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QRegularExpression>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>
#include <QtQml/qqmlregistration.h>

struct CatalogItem {
    qint64 skuId = 0;
    QString name;
    QString nameLower;
    QString brandLower;
    QString brand;
    double price = 0;
    QStringList tokens;
};

class SearchEngine : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    QML_SINGLETON

    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(int brandCount READ brandCount NOTIFY readyChanged)
    Q_PROPERTY(int categoryCount READ categoryCount NOTIFY readyChanged)
    Q_PROPERTY(int modelPhraseCount READ modelPhraseCount NOTIFY readyChanged)
    Q_PROPERTY(int catalogCount READ catalogCount NOTIFY readyChanged)
    Q_PROPERTY(QString appVersion READ appVersion CONSTANT)

public:
    explicit SearchEngine(QObject *parent = nullptr);

    bool ready() const { return m_ready; }
    int brandCount() const { return m_brandCanonical.size(); }
    int categoryCount() const { return m_categories.size(); }
    int modelPhraseCount() const { return m_modelPhrases.size(); }
    int catalogCount() const { return m_catalog.size(); }
    QString appVersion() const { return QStringLiteral("0.4.0"); }

    // извлечение фактов из запроса; возвращает объект с entities/brand/category/attributes/latency
    Q_INVOKABLE QVariantMap extract(const QString &query);

    // RecSys только по фактам: ранжирует каталог, возвращает топ-N карточек со скором
    Q_INVOKABLE QVariantList rankCatalog(const QVariantMap &facts, int topN = 30);

    Q_INVOKABLE QString extractJson(const QString &query);

    // запрос из переменной окружения MV_QUERY (utf-8) — для демо и скриншотов
    Q_INVOKABLE QString envQuery() const;
    Q_INVOKABLE QString envShootPath() const;

    // удаление приложения из настроек — как в ГК МОС: помечаем, что стереть, потом
    // запускаем деинсталлятор Inno Setup и закрываемся
    Q_INVOKABLE void prepareForUninstall(bool deleteSettings);
    Q_INVOKABLE void launchUninstaller();

signals:
    void readyChanged();

private:
    void loadData();
    QString dataDir() const;
    QStringList tokenize(const QString &text) const;
    static QString normalize(const QString &text);
    static QString splitGlued(const QString &text);
    QString markovType(const QStringList &spanTokens) const;

    // SpellFix v2: гомоглифы + алиасы транслита + fuzzy по словарю + опечатки единиц
    QString spellFixQuery(const QString &query) const;
    QString fixTokenSpell(const QString &token) const;
    QString normalizeHomoglyphs(const QString &token) const;
    QString bestFuzzyCanon(const QString &token) const;
    static int editDistance(const QString &a, const QString &b);
    static bool hasMixedScript(const QString &token);

    bool m_ready = false;

    // канонизация брендов: нижний регистр -> каноническое имя
    QHash<QString, QString> m_brandCanonical;
    QHash<QString, QString> m_brandAliases;
    QHash<QString, QString> m_spellAliases; // alias -> canon (сони→sony)
    QSet<QString> m_spellVocab;            // однословные бренды/категории/единицы
    QSet<QString> m_categories;
    QSet<QString> m_modelPhrases;   // фразы линеек из майнинга
    QSet<QString> m_colors;

    // марковский типизатор: bigram "a|b" -> тип, unit -> тип
    QHash<QString, QString> m_bigramType;
    QHash<QString, QString> m_unitType;

    // атрибутные регулярки: паттерн + тип
    QVector<QPair<QRegularExpression, QString>> m_attrPatterns;

    QVector<CatalogItem> m_catalog;
};
