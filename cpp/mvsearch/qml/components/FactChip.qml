// чип извлечённого факта: цветной бейдж типа + значение
import QtQuick
import MvSearch

Rectangle {
    id: chip

    property string tagLabel      // BRAND / CATEGORY / MODEL / ATTR
    property string tagRu         // бренд / категория / ...
    property string value
    property color tagColor: Theme.accent

    implicitWidth: row.implicitWidth + 26
    implicitHeight: 40
    radius: 20
    color: Theme.surface
    border.width: 1.4
    border.color: Qt.alpha(tagColor, 0.55)

    scale: 0
    Component.onCompleted: appearAnim.start()

    NumberAnimation {
        id: appearAnim
        target: chip; property: "scale"
        from: 0; to: 1; duration: 320
        easing.type: Easing.OutBack
        easing.overshoot: 1.4
    }

    Row {
        id: row
        anchors.centerIn: parent
        spacing: 8

        Rectangle {
            width: ruText.implicitWidth + 14
            height: 22
            radius: 11
            color: chip.tagColor
            anchors.verticalCenter: parent.verticalCenter

            Text {
                id: ruText
                anchors.centerIn: parent
                text: chip.tagRu
                font.pixelSize: 10
                font.family: Theme.fontFamily
                font.weight: Font.Bold
                font.capitalization: Font.AllUppercase
                color: "#ffffff"
            }
        }

        Text {
            text: chip.value
            font.pixelSize: Theme.fontBody
            font.family: Theme.fontFamily
            font.weight: Font.DemiBold
            color: Theme.text
            anchors.verticalCenter: parent.verticalCenter
        }
    }
}
