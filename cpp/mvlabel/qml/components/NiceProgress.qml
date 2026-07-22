// анимированный прогресс-бар с текстом
import QtQuick
import MvLabel

Item {
    id: root

    property int done: 0
    property int total: 1

    implicitHeight: 22

    Rectangle {
        anchors.fill: parent
        radius: height / 2
        color: Theme.surfaceAlt

        Rectangle {
            width: Math.max(height, parent.width * (root.done / Math.max(root.total, 1)))
            height: parent.height
            radius: height / 2
            color: Theme.accent

            Behavior on width { NumberAnimation { duration: 350; easing.type: Easing.OutCubic } }
        }

        Text {
            anchors.centerIn: parent
            text: root.done + " / " + root.total
            font.pixelSize: 10
            font.family: Theme.fontFamily
            font.weight: Font.Bold
            color: Theme.text
        }
    }
}
