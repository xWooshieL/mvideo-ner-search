// плитка со статистикой
import QtQuick
import MvSearch

AppCard {
    id: root

    property string label
    property string value
    property color valueColor: Theme.text

    implicitHeight: 84

    Column {
        anchors.verticalCenter: parent.verticalCenter
        anchors.left: parent.left
        anchors.leftMargin: 18
        anchors.right: parent.right
        anchors.rightMargin: 12
        spacing: 6

        Text {
            text: root.label
            font.pixelSize: Theme.fontSmall
            font.family: Theme.fontFamily
            color: Theme.textSecondary
            elide: Text.ElideRight
            width: parent.width
        }

        Text {
            text: root.value
            font.pixelSize: 22
            font.family: Theme.fontFamily
            font.weight: Font.Bold
            color: root.valueColor
            elide: Text.ElideRight
            width: parent.width
        }
    }
}
