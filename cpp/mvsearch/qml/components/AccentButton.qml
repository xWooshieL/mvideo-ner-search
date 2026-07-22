// основная кнопка с акцентным фоном и риплом от точки нажатия
import QtQuick
import QtQuick.Controls.Basic
import MvSearch

Button {
    id: control

    property bool danger: false

    focusPolicy: Qt.NoFocus

    implicitHeight: 40
    implicitWidth: Math.max(120, contentItem.implicitWidth + 40)

    font.pixelSize: Theme.fontBody
    font.family: Theme.fontFamily
    font.weight: Font.DemiBold

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? Theme.textOnAccent : Theme.textTertiary
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }

    background: Rectangle {
        id: backgroundRect
        radius: Theme.radiusSmall
        color: {
            if (!control.enabled)
                return Theme.surfaceAlt
            const base = control.danger ? Theme.error : Theme.accent
            const hovered = control.danger ? Qt.darker(Theme.error, 1.15) : Theme.accentHover
            return control.hovered ? hovered : base
        }

        Behavior on color { ColorAnimation { duration: 160; easing.type: Easing.OutCubic } }

        Item {
            anchors.fill: parent
            clip: true

            Rectangle {
                id: ripple
                width: 12
                height: 12
                radius: 6
                color: "#50ffffff"
                opacity: 0
                transformOrigin: Item.Center
            }
        }

        TapHandler {
            gesturePolicy: TapHandler.ReleaseWithinBounds
            onPressedChanged: {
                if (pressed && control.enabled) {
                    ripple.x = point.position.x - ripple.width / 2
                    ripple.y = point.position.y - ripple.height / 2
                    rippleAnimation.restart()
                }
            }
        }

        ParallelAnimation {
            id: rippleAnimation
            NumberAnimation {
                target: ripple; property: "scale"
                from: 1; to: Math.max(backgroundRect.width, 40) / 5
                duration: 420; easing.type: Easing.OutCubic
            }
            SequentialAnimation {
                NumberAnimation { target: ripple; property: "opacity"; from: 0; to: 0.9; duration: 80 }
                NumberAnimation { target: ripple; property: "opacity"; to: 0; duration: 340 }
            }
        }
    }

    HoverHandler { cursorShape: Qt.PointingHandCursor }
}
