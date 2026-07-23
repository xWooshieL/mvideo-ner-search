// М.Видео — Разметка, трёхэтапный мастер на C++/QML
#include <QApplication>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQuickStyle>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setOrganizationName(QStringLiteral("MVideo"));
    app.setApplicationName(QStringLiteral("М.Видео Разметка"));
    app.setApplicationVersion(QStringLiteral("0.1.2"));
    app.setWindowIcon(QIcon(QStringLiteral(":/qt/qml/MvLabel/assets/icon.png")));

    QQuickStyle::setStyle(QStringLiteral("Basic"));

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine, &QQmlApplicationEngine::objectCreationFailed,
        &app, []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);

    engine.loadFromModule("MvLabel", "Main");

    return app.exec();
}
