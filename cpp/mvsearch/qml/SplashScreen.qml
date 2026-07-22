// анимированное приветствие: эмблема «м» с пружинкой и мягким свечением
import QtQuick
import QtQuick.Effects
import MvSearch

Rectangle {
    id: splash

    signal finished()

    property bool introStarted: false

    color: Theme.bg

    function beginIntro() {
        if (introStarted)
            return
        introStarted = true
        introAnimation.start()
    }

    Timer {
        id: startTimer
        interval: 300
        running: true
        repeat: false
        onTriggered: splash.beginIntro()
    }

    // мягкое красное свечение за эмблемой
    Rectangle {
        id: glow
        width: 240
        height: 240
        radius: width / 2
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -30
        color: Theme.accent
        opacity: 0
        visible: false
    }

    MultiEffect {
        source: glow
        anchors.fill: glow
        anchors.margins: -80
        blurEnabled: true
        blur: 1.0
        blurMax: 150
        opacity: glow.opacity
    }

    Image {
        id: mark
        source: Theme.logoMarkSource
        width: 150
        height: 150
        fillMode: Image.PreserveAspectFit
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -30
        opacity: 0
        scale: 0.55
        rotation: -14
    }

    Text {
        id: titleText
        text: qsTr("М.ВИДЕО")
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: mark.bottom
        anchors.topMargin: 28
        font.pixelSize: 34
        font.family: Theme.fontFamily
        font.weight: Font.Bold
        font.letterSpacing: 6
        color: Theme.text
        opacity: 0
    }

    Text {
        id: subtitleText
        text: qsTr("Умный поиск — извлечение фактов из запроса")
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: titleText.bottom
        anchors.topMargin: 8
        font.pixelSize: Theme.fontMedium
        font.family: Theme.fontFamily
        color: Theme.textSecondary
        opacity: 0
    }

    Text {
        text: qsTr("версия %1").arg(SearchEngine.appVersion)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 32
        font.pixelSize: Theme.fontSmall
        font.family: Theme.fontFamily
        color: Theme.textTertiary
        opacity: introStarted ? subtitleText.opacity : 0
    }

    SequentialAnimation {
        id: introAnimation
        running: false

        // эмблема появляется с пружинкой и лёгким поворотом
        ParallelAnimation {
            NumberAnimation { target: mark; property: "opacity"; from: 0; to: 1; duration: 850; easing.type: Easing.OutCubic }
            NumberAnimation { target: mark; property: "scale"; from: 0.55; to: 1.0; duration: 1100; easing.type: Easing.OutBack; easing.overshoot: 1.25 }
            NumberAnimation { target: mark; property: "rotation"; from: -14; to: 0; duration: 1100; easing.type: Easing.OutCubic }
            NumberAnimation { target: glow; property: "opacity"; from: 0; to: Theme.dark ? 0.18 : 0.10; duration: 1200 }
        }

        ParallelAnimation {
            NumberAnimation { target: titleText; property: "opacity"; from: 0; to: 1; duration: 450 }
            NumberAnimation { target: titleText; property: "anchors.topMargin"; from: 44; to: 28; duration: 450; easing.type: Easing.OutCubic }
        }
        NumberAnimation { target: subtitleText; property: "opacity"; from: 0; to: 1; duration: 400 }

        NumberAnimation { target: glow; property: "opacity"; to: Theme.dark ? 0.24 : 0.14; duration: 500; easing.type: Easing.InOutSine }

        PauseAnimation { duration: 550 }

        ScriptAction { script: splash.finished() }
    }

    TapHandler {
        onTapped: {
            if (!introStarted) {
                startTimer.stop()
                splash.beginIntro()
            } else {
                splash.finished()
            }
        }
    }
}
