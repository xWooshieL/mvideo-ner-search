// хранилище разметки: JSONL-файлы рядом с exe, аннотатор из mvlabel.ini или имени папки
#include "labelstore.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QGuiApplication>
#include <QJsonDocument>
#include <QProcess>
#include <QSettings>
#include <QStandardPaths>
#include <QTextStream>
#include <QTimer>
#include <QUrl>

LabelStore::LabelStore(QObject *parent)
    : QObject(parent)
{
    load();
}

QString LabelStore::dataDir() const
{
    // рядом с exe (Windows), в Resources бандла (macOS .app) либо на уровень выше (dev-сборка)
    const QString appDir = QCoreApplication::applicationDirPath();
    for (const QString &candidate : {appDir + u"/data"_qs,
                                     appDir + u"/../Resources/data"_qs,
                                     appDir + u"/../data"_qs}) {
        QDir d(candidate);
        if (!d.entryList({u"queries_*.json"_qs}, QDir::Files).isEmpty())
            return d.absolutePath();
    }
    return appDir + u"/data"_qs;
}

QString LabelStore::labelsDir() const
{
    // Windows/dev: labels рядом с exe. macOS: .app read-only после подписи/дистрибуции —
    // пишем в стандартную папку данных пользователя, чтобы разметка не терялась.
#ifdef Q_OS_MACOS
    QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + u"/labels"_qs;
#else
    QString dir = QCoreApplication::applicationDirPath() + u"/labels"_qs;
#endif
    QDir().mkpath(dir);
    return dir;
}

QString LabelStore::iniPath() const
{
#ifdef Q_OS_MACOS
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dir);
    return dir + u"/mvlabel.ini"_qs;
#else
    return QCoreApplication::applicationDirPath() + u"/mvlabel.ini"_qs;
#endif
}

void LabelStore::load()
{
    m_queries.clear();
    m_pairs.clear();
    m_bioRecords.clear();
    m_matchRecords.clear();

    // аннотатор: явный выбор -> аргумент запуска -> mvlabel.ini -> env -> nikita
    m_annotatorKey = m_overrideKey;
    if (m_annotatorKey.isEmpty()) {
        const QStringList args = QCoreApplication::arguments();
        for (const QString &a : args) {
            const QString low = a.toLower();
            if (low == u"nikita"_qs || low == u"nekit"_qs || low == u"liza"_qs) {
                m_annotatorKey = low;
                break;
            }
        }
    }
    if (m_annotatorKey.isEmpty()) {
        QSettings ini(iniPath(), QSettings::IniFormat);
        m_annotatorKey = ini.value(u"annotator"_qs).toString();
    }
    if (m_annotatorKey.isEmpty())
        m_annotatorKey = qEnvironmentVariable("MV_ANNOTATOR", u"nikita"_qs);

    const QString dir = dataDir();

    QFile qf(dir + QStringLiteral("/queries_%1.json").arg(m_annotatorKey));
    if (qf.open(QIODevice::ReadOnly)) {
        const QJsonObject root = QJsonDocument::fromJson(qf.readAll()).object();
        m_annotatorDisplay = root.value(u"annotator"_qs).toString(m_annotatorKey);
        for (const QJsonValue &v : root.value(u"queries"_qs).toArray())
            m_queries << v.toString();
    }

    QFile pf(dir + QStringLiteral("/pairs_%1.json").arg(m_annotatorKey));
    if (pf.open(QIODevice::ReadOnly)) {
        const QJsonObject root = QJsonDocument::fromJson(pf.readAll()).object();
        for (const QJsonValue &v : root.value(u"pairs"_qs).toArray())
            m_pairs << v.toObject().toVariantMap();
    }

    // существующая разметка
    auto readJsonl = [](const QString &path, QMap<int, QJsonObject> &out) {
        QFile f(path);
        if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
            return;
        QTextStream st(&f);
        st.setEncoding(QStringConverter::Utf8);
        while (!st.atEnd()) {
            const QJsonObject o = QJsonDocument::fromJson(st.readLine().toUtf8()).object();
            if (o.contains(u"index"_qs))
                out.insert(o.value(u"index"_qs).toInt(), o);
        }
    };
    readJsonl(labelsDir() + QStringLiteral("/bio_%1.jsonl").arg(m_annotatorKey), m_bioRecords);
    readJsonl(labelsDir() + QStringLiteral("/match_%1.jsonl").arg(m_annotatorKey), m_matchRecords);

    m_ready = !m_queries.isEmpty();
    emit loaded();
    emit bioSaved();
    emit matchSaved();
}

int LabelStore::firstUnlabeledBio() const
{
    for (int i = 0; i < m_queries.size(); ++i)
        if (!m_bioRecords.contains(i))
            return i;
    return 0;
}

int LabelStore::firstUnlabeledMatch() const
{
    for (int i = 0; i < m_pairs.size(); ++i)
        if (!m_matchRecords.contains(i))
            return i;
    return 0;
}

QVariantMap LabelStore::bioRecord(int index) const
{
    return m_bioRecords.value(index).toVariantMap();
}

QVariantMap LabelStore::matchRecord(int index) const
{
    return m_matchRecords.value(index).toVariantMap();
}

void LabelStore::saveBio(int index, const QString &query,
                         const QStringList &tags, const QVariantMap &subtypes)
{
    QJsonObject rec;
    rec[u"index"_qs] = index;
    rec[u"query"_qs] = query;
    rec[u"tags"_qs] = QJsonArray::fromStringList(tags);
    rec[u"subtypes"_qs] = QJsonObject::fromVariantMap(subtypes);
    rec[u"annotator"_qs] = m_annotatorDisplay;
    rec[u"ts"_qs] = QDateTime::currentDateTime().toString(Qt::ISODate);
    m_bioRecords.insert(index, rec);
    flushBio();
    emit bioSaved();
}

void LabelStore::saveMatch(int index, const QVariantMap &pair, int label, bool autoLabel)
{
    QJsonObject rec;
    rec[u"index"_qs] = index;
    rec[u"query"_qs] = pair.value(u"query"_qs).toString();
    rec[u"sku_name"_qs] = pair.value(u"sku_name"_qs).toString();
    rec[u"brand"_qs] = pair.value(u"brand"_qs).toString();
    rec[u"label"_qs] = label;
    rec[u"auto"_qs] = autoLabel;
    rec[u"annotator"_qs] = m_annotatorDisplay;
    rec[u"ts"_qs] = QDateTime::currentDateTime().toString(Qt::ISODate);
    m_matchRecords.insert(index, rec);
    flushMatch();
    emit matchSaved();
}

void LabelStore::flushBio()
{
    QFile f(labelsDir() + QStringLiteral("/bio_%1.jsonl").arg(m_annotatorKey));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
    QTextStream st(&f);
    st.setEncoding(QStringConverter::Utf8);
    for (auto it = m_bioRecords.begin(); it != m_bioRecords.end(); ++it)
        st << QJsonDocument(it.value()).toJson(QJsonDocument::Compact) << '\n';
}

void LabelStore::flushMatch()
{
    QFile f(labelsDir() + QStringLiteral("/match_%1.jsonl").arg(m_annotatorKey));
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
    QTextStream st(&f);
    st.setEncoding(QStringConverter::Utf8);
    for (auto it = m_matchRecords.begin(); it != m_matchRecords.end(); ++it)
        st << QJsonDocument(it.value()).toJson(QJsonDocument::Compact) << '\n';
}

void LabelStore::openLabelsFolder()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(labelsDir()));
}

void LabelStore::setAnnotator(const QString &key)
{
    const QString low = key.toLower();
    if (low == m_annotatorKey)
        return;
    m_overrideKey = low;
    // запоминаем выбор на следующие запуски
    QSettings ini(iniPath(), QSettings::IniFormat);
    ini.setValue(u"annotator"_qs, low);
    load();
}

QString LabelStore::envValue(const QString &name) const
{
    return qEnvironmentVariable(name.toUtf8().constData());
}

QVariantList LabelStore::bioHistory(int limit) const
{
    QVariantList out;
    for (auto it = m_bioRecords.constEnd(); it != m_bioRecords.constBegin() && out.size() < limit;) {
        --it;
        out.append(it.value().toVariantMap());
    }
    return out;
}

QVariantList LabelStore::matchHistory(int limit) const
{
    QVariantList out;
    for (auto it = m_matchRecords.constEnd(); it != m_matchRecords.constBegin() && out.size() < limit;) {
        --it;
        out.append(it.value().toVariantMap());
    }
    return out;
}

void LabelStore::prepareForUninstall(bool deleteLabels, bool deleteSettings)
{
    if (deleteLabels) {
        QDir labels(labelsDir());
        if (labels.exists())
            labels.removeRecursively();
    }

    if (deleteSettings) {
        QFile::remove(iniPath());
        QSettings settings(QStringLiteral("MVideo"), QStringLiteral("MvLabel"));
        settings.clear();
        settings.sync();
    }

    /* строки "1"/"0" — деинсталлятор читает RegQueryStringValue, не DWORD */
    QSettings uninstallFlags(QStringLiteral("MVideo"), QStringLiteral("UninstallMvLabel"));
    uninstallFlags.setValue(
        QStringLiteral("DeleteLabels"),
        deleteLabels ? QStringLiteral("1") : QStringLiteral("0"));
    uninstallFlags.setValue(
        QStringLiteral("DeleteSettings"),
        deleteSettings ? QStringLiteral("1") : QStringLiteral("0"));
    uninstallFlags.sync();
}

void LabelStore::launchUninstaller()
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
