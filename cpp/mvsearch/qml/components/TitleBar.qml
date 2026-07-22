// кастомная шапка окна: логотип, название, кнопки свернуть/развернуть/закрыть
import QtQuick
import MvSearch

Rectangle {
    id: titleBar

    required property Window targetWindow

    height: 38
    color: Theme.surface

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width
        height: 1
        color: Theme.border
    }

    DragHandler {
        target: null
        onActiveChanged: {
            if (active)
                titleBar.targetWindow.startSystemMove()
        }
    }

    TapHandler {
        onDoubleTapped: titleBar.toggleMaximize()
    }

    function toggleMaximize() {
        if (typeof targetWindow.toggleMaximizeAnimated === "function")
            targetWindow.toggleMaximizeAnimated()
        else if (targetWindow.frameMaximized)
            targetWindow.showNormal()
        else
            targetWindow.showMaximized()
    }

    Row {
        anchors.left: parent.left
        anchors.leftMargin: 12
        anchors.verticalCenter: parent.verticalCenter
        spacing: 9

        Image {
            source: Theme.logoMarkSource
            width: 20
            height: 20
            sourceSize.width: 40
            sourceSize.height: 40
            fillMode: Image.PreserveAspectFit
            anchors.verticalCenter: parent.verticalCenter
            asynchronous: false
        }

        Text {
            text: titleBar.targetWindow.title
            font.pixelSize: 12
            font.family: Theme.fontFamily
            font.weight: Font.Medium
            color: Theme.textSecondary
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    Row {
        anchors.right: parent.right
        anchors.top: parent.top
        height: parent.height

        component WindowButton : Rectangle {
            id: winButton

            property string glyph
            property bool closeButton: false
            signal activated()

            width: 46
            height: titleBar.height
            color: hoverHandler.hovered
                   ? (closeButton ? "#e81123" : Theme.hover)
                   : "transparent"

            Behavior on color { ColorAnimation { duration: 130 } }

            Text {
                anchors.centerIn: parent
                text: winButton.glyph
                font.pixelSize: 13
                font.family: "Segoe MDL2 Assets"
                color: hoverHandler.hovered && winButton.closeButton ? "#ffffff" : Theme.textSecondary
            }

            HoverHandler { id: hoverHandler }
            TapHandler { onTapped: winButton.activated() }
        }

        WindowButton {
            glyph: "\uE921"
            onActivated: {
                if (typeof titleBar.targetWindow.animatedMinimize === "function")
                    titleBar.targetWindow.animatedMinimize()
                else
                    titleBar.targetWindow.showMinimized()
            }
        }

        WindowButton {
            glyph: titleBar.targetWindow.frameMaximized ? "\uE923" : "\uE922"
            onActivated: titleBar.toggleMaximize()
        }

        WindowButton {
            glyph: "\uE8BB"
            closeButton: true
            onActivated: titleBar.targetWindow.close()
        }
    }
}
