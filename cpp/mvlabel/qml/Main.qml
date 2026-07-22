// корневое окно приложения разметки: frameless-рамка + сплэш -> главное окно
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import MvLabel
import "components"

ApplicationWindow {
    id: window

    width: 1180
    height: 820
    minimumWidth: 1000
    minimumHeight: 680
    visible: true
    title: qsTr("М.Видео — Разметка · %1").arg(LabelStore.annotator)
    color: "transparent"
    flags: Qt.Window | Qt.FramelessWindowHint

    property bool frameMaximized: false
    property rect normalGeometry: Qt.rect(x, y, width, height)

    Component.onCompleted: {
        const area = screen ? screen.availableGeometry : Qt.rect(0, 0, 1920, 1080)
        x = area.x + Math.max(0, Math.round((area.width - width) / 2))
        y = area.y + Math.max(0, Math.round((area.height - height) / 2))
    }

    function toggleMaximizeAnimated() {
        if (frameMaximized) {
            frameMaximized = false
            x = normalGeometry.x; y = normalGeometry.y
            width = normalGeometry.width; height = normalGeometry.height
        } else {
            normalGeometry = Qt.rect(x, y, width, height)
            const area = screen.availableGeometry
            frameMaximized = true
            x = area.x; y = area.y
            width = area.width; height = area.height
        }
    }

    function animatedMinimize() {
        window.showMinimized()
    }

    Rectangle {
        id: windowFrame
        anchors.fill: parent
        color: Theme.bg
        radius: window.frameMaximized ? 0 : 8
        border.width: window.frameMaximized ? 0 : 1
        border.color: Theme.border
        clip: true

        TitleBar {
            id: titleBar
            targetWindow: window
            anchors.top: parent.top
            anchors.left: parent.left
            anchors.right: parent.right
        }

        StackView {
            id: stack
            anchors.top: titleBar.bottom
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            initialItem: splashComponent

            replaceEnter: Transition {
                NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 350; easing.type: Easing.OutCubic }
            }
            replaceExit: Transition {
                NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 250 }
            }
        }
    }

    component ResizeEdge : MouseArea {
        property int edges
        hoverEnabled: false
        acceptedButtons: Qt.LeftButton
        onPressed: window.startSystemResize(edges)
        visible: !window.frameMaximized
    }

    ResizeEdge { anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom; width: 5; edges: Qt.LeftEdge; cursorShape: Qt.SizeHorCursor }
    ResizeEdge { anchors.right: parent.right; anchors.top: parent.top; anchors.bottom: parent.bottom; width: 5; edges: Qt.RightEdge; cursorShape: Qt.SizeHorCursor }
    ResizeEdge { anchors.top: parent.top; anchors.left: parent.left; anchors.right: parent.right; height: 5; edges: Qt.TopEdge; cursorShape: Qt.SizeVerCursor }
    ResizeEdge { anchors.bottom: parent.bottom; anchors.left: parent.left; anchors.right: parent.right; height: 5; edges: Qt.BottomEdge; cursorShape: Qt.SizeVerCursor }

    Component {
        id: splashComponent
        SplashScreen {
            onFinished: stack.replace(mainComponent)
        }
    }

    Component {
        id: mainComponent
        MainWindow { }
    }

    // самосъёмка для скриншотов: MV_SHOOT=<файл>
    Timer {
        interval: 6500
        running: LabelStore.envValue("MV_SHOOT").length > 0
        repeat: false
        onTriggered: {
            windowFrame.grabToImage(function(result) {
                result.saveToFile(LabelStore.envValue("MV_SHOOT"))
                Qt.quit()
            })
        }
    }
}
