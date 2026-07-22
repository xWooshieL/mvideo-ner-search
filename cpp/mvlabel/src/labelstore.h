// хранилище разметки: загрузка данных аннотатора, сохранение BIO и match 1/0 в JSONL
#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QString>
#include <QVariantList>
#include <QVariantMap>
#include <QtQml/qqmlregistration.h>

class LabelStore : public QObject
{
    Q_OBJECT
    QML_ELEMENT
    QML_SINGLETON

    Q_PROPERTY(QString annotator READ annotator NOTIFY loaded)
    Q_PROPERTY(QString annotatorId READ annotatorKey NOTIFY loaded)
    Q_PROPERTY(QStringList queries READ queries NOTIFY loaded)
    Q_PROPERTY(QVariantList pairs READ pairs NOTIFY loaded)
    Q_PROPERTY(int bioDone READ bioDone NOTIFY bioSaved)
    Q_PROPERTY(int matchDone READ matchDone NOTIFY matchSaved)
    Q_PROPERTY(bool ready READ ready NOTIFY loaded)

public:
    explicit LabelStore(QObject *parent = nullptr);

    QString annotator() const { return m_annotatorDisplay; }
    QStringList queries() const { return m_queries; }
    QVariantList pairs() const { return m_pairs; }
    int bioDone() const { return m_bioRecords.size(); }
    int matchDone() const { return m_matchRecords.size(); }
    bool ready() const { return m_ready; }

    // первый неразмеченный индекс
    Q_INVOKABLE int firstUnlabeledBio() const;
    Q_INVOKABLE int firstUnlabeledMatch() const;

    Q_INVOKABLE QVariantMap bioRecord(int index) const;
    Q_INVOKABLE QVariantMap matchRecord(int index) const;

    // сохранение: tags — список строк "B-BRAND"/"O", subtypes — map индекс->подтип
    Q_INVOKABLE void saveBio(int index, const QString &query,
                             const QStringList &tags, const QVariantMap &subtypes);
    Q_INVOKABLE void saveMatch(int index, const QVariantMap &pair, int label, bool autoLabel);

    Q_INVOKABLE void openLabelsFolder();

    // переключение аккаунта без перезапуска (один exe на команду)
    QString annotatorKey() const { return m_annotatorKey; }
    Q_INVOKABLE void setAnnotator(const QString &key);

    // env-переменные для демо и скриншотов
    Q_INVOKABLE QString envValue(const QString &name) const;

    Q_INVOKABLE QVariantList bioHistory(int limit = 200) const;
    Q_INVOKABLE QVariantList matchHistory(int limit = 200) const;

signals:
    void loaded();
    void bioSaved();
    void matchSaved();

private:
    void load();
    QString dataDir() const;
    QString labelsDir() const;
    void flushBio();
    void flushMatch();

    bool m_ready = false;
    QString m_overrideKey;
    QString m_annotatorKey;
    QString m_annotatorDisplay;
    QStringList m_queries;
    QVariantList m_pairs;

    QMap<int, QJsonObject> m_bioRecords;
    QMap<int, QJsonObject> m_matchRecords;
};
