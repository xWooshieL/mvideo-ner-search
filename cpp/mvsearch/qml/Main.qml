// корневое окно: кастомная frameless-рамка + сплэш -> главное окно
import QtQuick
import QtQuick.Window
import QtQuick.Controls.Basic
import MvSearch
import "components"

ApplicationWindow {
    id: window

    property int windowWidth: 1280
    property int windowHeight: 840

    width: windowWidth
    height: windowHeight
    minimumWidth: 1080
    minimumHeight: 700
    visible: true
    title: qsTr("М.Видео — Умный поиск")
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
        minimizeAnimation.restart()
    }

    SequentialAnimation {
        id: minimizeAnimation
        ParallelAnimation {
            NumberAnimation { target: windowFrame; property: "opacity"; to: 0; duration: 180; easing.type: Easing.InCubic }
            NumberAnimation { target: windowFrame; property: "scale"; to: 0.96; duration: 200; easing.type: Easing.InOutCubic }
        }
        ScriptAction {
            script: {
                window.showMinimized()
                windowFrame.opacity = 1
                windowFrame.scale = 1
            }
        }
    }

    property int previousVisibility: Window.Windowed
    onVisibilityChanged: {
        if (previousVisibility === Window.Minimized
            && (visibility === Window.Windowed || visibility === Window.Maximized)) {
            windowFrame.opacity = 0
            windowFrame.scale = 0.97
            restoreAnimation.restart()
        }
        previousVisibility = visibility
    }

    ParallelAnimation {
        id: restoreAnimation
        NumberAnimation { target: windowFrame; property: "opacity"; from: 0; to: 1; duration: 260; easing.type: Easing.OutCubic }
        NumberAnimation { target: windowFrame; property: "scale"; from: 0.97; to: 1; duration: 280; easing.type: Easing.OutCubic }
    }

    Rectangle {
        id: windowFrame
        anchors.fill: parent
        color: Theme.bg
        radius: window.frameMaximized ? 0 : 8
        border.width: window.frameMaximized ? 0 : 1
        border.color: Theme.border
        clip: true
        transformOrigin: Item.Center

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

            pushEnter: Transition {
                NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 350; easing.type: Easing.OutCubic }
            }
            pushExit: Transition {
                NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 250 }
            }
            replaceEnter: Transition {
                NumberAnimation { property: "opacity"; from: 0; to: 1; duration: 350; easing.type: Easing.OutCubic }
            }
            replaceExit: Transition {
                NumberAnimation { property: "opacity"; from: 1; to: 0; duration: 250 }
            }
        }

        Toast {
            id: globalToast
            anchors.top: titleBar.bottom
            anchors.right: parent.right
            anchors.rightMargin: 18
            anchors.topMargin: 18
            z: 100
        }
    }

    // ресайз за края frameless-окна
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
    ResizeEdge { anchors.left: parent.left; anchors.top: parent.top; width: 10; height: 10; edges: Qt.LeftEdge | Qt.TopEdge; cursorShape: Qt.SizeFDiagCursor }
    ResizeEdge { anchors.right: parent.right; anchors.top: parent.top; width: 10; height: 10; edges: Qt.RightEdge | Qt.TopEdge; cursorShape: Qt.SizeBDiagCursor }
    ResizeEdge { anchors.left: parent.left; anchors.bottom: parent.bottom; width: 10; height: 10; edges: Qt.LeftEdge | Qt.BottomEdge; cursorShape: Qt.SizeBDiagCursor }
    ResizeEdge { anchors.right: parent.right; anchors.bottom: parent.bottom; width: 10; height: 10; edges: Qt.RightEdge | Qt.BottomEdge; cursorShape: Qt.SizeFDiagCursor }

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

    // самосъёмка для скриншотов: MV_SHOOT=<файл> — окно само сохраняет кадр и выходит
    Timer {
        interval: 6500
        running: SearchEngine.envShootPath().length > 0
        repeat: false
        onTriggered: {
            windowFrame.grabToImage(function(result) {
                result.saveToFile(SearchEngine.envShootPath())
                Qt.quit()
            })
        }
    }
}
