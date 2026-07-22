// пункт бокового меню: активная полоска слева, плавные ховеры
import QtQuick
import MvSearch

Rectangle {
    id: root

    property string label
    property string icon
    property bool active: false
    signal clicked()

    height: 42
    radius: Theme.radiusSmall
    color: active ? Theme.accentSoft : "transparent"

    Behavior on color { ColorAnimation { duration: 280; easing.type: Easing.InOutCubic } }

    Rectangle {
        id: hoverFill
        anchors.fill: parent
        radius: root.radius
        color: Theme.sidebarHover
        opacity: hoverArea.hovered && !root.active
               ? (Theme.dark ? 0.12 : 1.0)
               : 0.0
        visible: !root.active

        Behavior on opacity {
            NumberAnimation {
                duration: hoverArea.hovered ? 90 : 280
                easing.type: Easing.InOutCubic
            }
        }
    }

    Rectangle {
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        width: 3
        height: root.active ? 20 : 0
        radius: 1.5
        color: Theme.accent
        z: 1

        Behavior on height { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
    }

    Row {
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 14
        spacing: 10
        z: 1

        Item {
            width: 22
            height: 22
            anchors.verticalCenter: parent.verticalCenter

            Text {
                anchors.centerIn: parent
                text: root.icon
                font.pixelSize: 15
                font.family: Theme.iconFont
                color: root.active ? Theme.accent : Theme.textSecondary

                Behavior on color { ColorAnimation { duration: 320; easing.type: Easing.InOutCubic } }
            }
        }

        Text {
            text: root.label
            font.pixelSize: Theme.fontBody
            font.family: Theme.fontFamily
            font.weight: root.active ? Font.DemiBold : Font.Medium
            color: root.active ? Theme.accent : Theme.text
            anchors.verticalCenter: parent.verticalCenter

            Behavior on color { ColorAnimation { duration: 320; easing.type: Easing.InOutCubic } }
        }
    }

    HoverHandler {
        id: hoverArea
        cursorShape: Qt.PointingHandCursor
    }

    TapHandler {
        onTapped: root.clicked()
    }
}
