// текстовое поле в стиле приложения
import QtQuick
import QtQuick.Controls.Basic
import MvSearch

TextField {
    id: control

    implicitHeight: 42
    font.pixelSize: Theme.fontBody
    font.family: Theme.fontFamily
    color: Theme.text
    placeholderTextColor: Theme.textTertiary
    selectByMouse: true
    leftPadding: 14
    rightPadding: 14

    background: Rectangle {
        radius: Theme.radiusSmall
        color: Theme.surfaceAlt
        border.width: control.activeFocus ? 2 : 1
        border.color: control.activeFocus ? Theme.accent : Theme.border

        Behavior on border.color { ColorAnimation { duration: 120 } }
    }
}
