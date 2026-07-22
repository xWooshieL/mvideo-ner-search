// вторичная кнопка с рамкой
import QtQuick
import QtQuick.Controls.Basic
import MvLabel

Button {
    id: control

    focusPolicy: Qt.NoFocus

    implicitHeight: 40
    implicitWidth: Math.max(100, contentItem.implicitWidth + 36)

    font.pixelSize: Theme.fontBody
    font.family: Theme.fontFamily
    font.weight: Font.Medium

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? Theme.text : Theme.textTertiary
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        radius: Theme.radiusSmall
        color: "transparent"
        border.width: 1
        border.color: control.hovered ? Theme.borderStrong : Theme.border

        Behavior on border.color { ColorAnimation { duration: 120 } }

        Rectangle {
            anchors.fill: parent
            anchors.margins: 1
            radius: Theme.radiusSmall - 1
            color: Theme.sidebarHover
            opacity: control.hovered || control.down
                   ? (Theme.dark ? 0.12 : 1.0)
                   : 0.0

            Behavior on opacity {
                NumberAnimation {
                    duration: control.hovered ? 100 : 220
                    easing.type: Easing.InOutCubic
                }
            }
        }
    }

    HoverHandler { cursorShape: Qt.PointingHandCursor }
}

