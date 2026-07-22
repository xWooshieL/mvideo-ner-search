// палитра М.Видео для приложения разметки
pragma Singleton
import QtQuick

QtObject {
    id: theme

    property bool dark: false

    readonly property color bg: "#faf6f6"
    readonly property color surface: "#ffffff"
    readonly property color surfaceAlt: "#f6eeee"
    readonly property color hover: "#f3e5e5"

    readonly property color border: "#eddede"
    readonly property color borderStrong: "#dcc2c2"

    readonly property color text: "#26181a"
    readonly property color textSecondary: "#7a6467"
    readonly property color textTertiary: "#ab989a"

    readonly property color accent: "#f20601"
    readonly property color accentHover: "#cc0500"
    readonly property color accentSoft: "#fdecec"
    readonly property color textOnAccent: "#ffffff"

    readonly property color success: "#18794e"
    readonly property color successSoft: "#e0f2e9"
    readonly property color warning: "#b47d10"
    readonly property color error: "#c53a31"
    readonly property color errorSoft: "#fbe4e2"

    // цвета типов сущностей
    readonly property color tagBrand: "#1c1c1e"
    readonly property color tagModel: "#1f8a50"
    readonly property color tagCategory: "#f20601"
    readonly property color tagAttr: "#c75000"
    readonly property color tagGenre: "#6a35b0"
    readonly property color tagO: "#8e8e93"

    readonly property int radiusSmall: 6
    readonly property int radiusMedium: 10
    readonly property int radiusLarge: 16

    readonly property int fontSmall: 12
    readonly property int fontBody: 13
    readonly property int fontMedium: 15
    readonly property int fontLarge: 19
    readonly property int fontTitle: 24

    readonly property bool isWindows: Qt.platform.os === "windows"
    readonly property string fontFamily: isWindows ? "Segoe UI"
                                      : (Qt.platform.os === "osx" || Qt.platform.os === "macos")
                                        ? ".AppleSystemUIFont" : "sans-serif"
    readonly property string iconFont: isWindows ? "Segoe MDL2 Assets" : fontFamily
    readonly property string iconMinimize: isWindows ? "\uE921" : "\u2212"
    readonly property string iconMaximize: isWindows ? "\uE922" : "\u25AD"
    readonly property string iconRestore:  isWindows ? "\uE923" : "\u229E"
    readonly property string iconClose:    isWindows ? "\uE8BB" : "\u00D7"

    readonly property color sidebarHover: "#faf2f2"

    readonly property url logoMarkSource: "qrc:/qt/qml/MvLabel/assets/logo_mark.png"
}
