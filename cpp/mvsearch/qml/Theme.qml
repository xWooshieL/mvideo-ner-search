// палитра М.Видео: красно-белый цветокор, светлая и тёмная тема
pragma Singleton
import QtQuick

QtObject {
    id: theme

    property bool dark: false

    // фоны
    readonly property color bg: dark ? "#16090a" : "#faf6f6"
    readonly property color surface: dark ? "#211013" : "#ffffff"
    readonly property color surfaceAlt: dark ? "#2a1518" : "#f6eeee"
    readonly property color hover: dark ? "#33191d" : "#f3e5e5"

    // границы
    readonly property color border: dark ? "#3a1f24" : "#eddede"
    readonly property color borderStrong: dark ? "#5c2f37" : "#dcc2c2"

    // текст
    readonly property color text: dark ? "#f4e8e8" : "#26181a"
    readonly property color textSecondary: dark ? "#b39a9c" : "#7a6467"
    readonly property color textTertiary: dark ? "#7d6467" : "#ab989a"

    // акцент — фирменный красный М.Видео
    readonly property color accent: "#f20601"
    readonly property color accentHover: dark ? "#ff3630" : "#cc0500"
    readonly property color accentSoft: dark ? "#42191b" : "#fdecec"
    readonly property color textOnAccent: "#ffffff"

    // статусы
    readonly property color success: dark ? "#4cc38a" : "#18794e"
    readonly property color successSoft: dark ? "#1d3229" : "#e0f2e9"
    readonly property color warning: dark ? "#e5a13c" : "#b47d10"
    readonly property color error: dark ? "#e5665c" : "#c53a31"
    readonly property color errorSoft: dark ? "#3a2020" : "#fbe4e2"

    // цвета тегов сущностей
    readonly property color tagBrand: dark ? "#e8ecf4" : "#1c1c1e"
    readonly property color tagCategory: "#f20601"
    readonly property color tagModel: "#1f8a50"
    readonly property color tagAttr: "#c75000"
    readonly property color tagGenre: "#6a35b0"

    // размеры
    readonly property int radiusSmall: 6
    readonly property int radiusMedium: 10
    readonly property int radiusLarge: 16

    readonly property int fontSmall: 12
    readonly property int fontBody: 13
    readonly property int fontMedium: 15
    readonly property int fontLarge: 19
    readonly property int fontTitle: 24

    readonly property string fontFamily: "Segoe UI"

    readonly property color sidebarHover: dark ? "#ffffff" : "#faf2f2"

    readonly property url logoMarkSource: dark
        ? "qrc:/qt/qml/MvSearch/assets/logo_mark_white.png"
        : "qrc:/qt/qml/MvSearch/assets/logo_mark.png"
}
