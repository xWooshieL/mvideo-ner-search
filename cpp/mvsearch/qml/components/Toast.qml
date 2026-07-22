// всплывающее уведомление справа сверху
import QtQuick
import MvSearch

Item {
    id: toast

    property string message
    property bool isError: true

    width: 360
    height: card.height
    opacity: 0
    visible: opacity > 0

    function show(text) {
        message = text
        hideTimer.restart()
        appear.restart()
    }

    ParallelAnimation {
        id: appear
        NumberAnimation {
            target: toast; property: "opacity"
            from: 0; to: 1; duration: 280; easing.type: Easing.OutCubic
        }
        NumberAnimation {
            target: toast; property: "anchors.topMargin"
            from: 6; to: 18; duration: 320; easing.type: Easing.OutCubic
        }
    }

    NumberAnimation {
        id: disappear
        target: toast; property: "opacity"
        to: 0; duration: 350; easing.type: Easing.InCubic
    }

    Timer {
        id: hideTimer
        interval: 5000
        onTriggered: disappear.start()
    }

    Rectangle {
        id: card
        width: parent.width
        height: contentRow.implicitHeight + 26
        radius: Theme.radiusMedium
        color: toast.isError ? Theme.errorSoft : Theme.surface
        border.width: 1
        border.color: toast.isError ? "#f3c8c4" : Theme.border

        Row {
            id: contentRow
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.leftMargin: 16
            anchors.rightMargin: 40
            spacing: 12

            Rectangle {
                width: 30
                height: 30
                radius: 15
                color: toast.isError ? Theme.error : Theme.accent
                anchors.verticalCenter: parent.verticalCenter

                Text {
                    anchors.centerIn: parent
                    text: toast.isError ? "!" : "✓"
                    font.pixelSize: 16
                    font.weight: Font.Bold
                    color: "#ffffff"
                }
            }

            Column {
                spacing: 2
                width: parent.width - 42
                anchors.verticalCenter: parent.verticalCenter

                Text {
                    text: toast.isError ? qsTr("Ошибка") : qsTr("Готово")
                    font.pixelSize: Theme.fontBody
                    font.family: Theme.fontFamily
                    font.weight: Font.DemiBold
                    color: Theme.text
                }

                Text {
                    text: toast.message
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                    wrapMode: Text.WordWrap
                    width: parent.width
                }
            }
        }

        Text {
            text: "✕"
            anchors.top: parent.top
            anchors.right: parent.right
            anchors.margins: 12
            font.pixelSize: 12
            color: closeHover.hovered ? Theme.text : Theme.textTertiary

            Behavior on color { ColorAnimation { duration: 130 } }

            HoverHandler { id: closeHover; cursorShape: Qt.PointingHandCursor }
            TapHandler { onTapped: disappear.start() }
        }
    }
}
