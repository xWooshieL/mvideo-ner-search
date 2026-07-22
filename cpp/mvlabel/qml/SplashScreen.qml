// сплэш приложения разметки: та же анимация «м» с пружинкой
import QtQuick
import QtQuick.Effects
import MvLabel

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
        interval: 250
        running: true
        repeat: false
        onTriggered: splash.beginIntro()
    }

    Rectangle {
        id: glow
        width: 220
        height: 220
        radius: width / 2
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -26
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
        width: 130
        height: 130
        fillMode: Image.PreserveAspectFit
        anchors.centerIn: parent
        anchors.verticalCenterOffset: -26
        opacity: 0
        scale: 0.55
        rotation: -14
    }

    Text {
        id: titleText
        text: qsTr("РАЗМЕТКА")
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: mark.bottom
        anchors.topMargin: 26
        font.pixelSize: 30
        font.family: Theme.fontFamily
        font.weight: Font.Bold
        font.letterSpacing: 6
        color: Theme.text
        opacity: 0
    }

    Text {
        id: subtitleText
        text: qsTr("Золотая BIO-разметка и соответствие 1/0 · %1").arg(LabelStore.annotator)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: titleText.bottom
        anchors.topMargin: 8
        font.pixelSize: Theme.fontMedium
        font.family: Theme.fontFamily
        color: Theme.textSecondary
        opacity: 0
    }

    SequentialAnimation {
        id: introAnimation
        running: false

        ParallelAnimation {
            NumberAnimation { target: mark; property: "opacity"; from: 0; to: 1; duration: 800; easing.type: Easing.OutCubic }
            NumberAnimation { target: mark; property: "scale"; from: 0.55; to: 1.0; duration: 1050; easing.type: Easing.OutBack; easing.overshoot: 1.25 }
            NumberAnimation { target: mark; property: "rotation"; from: -14; to: 0; duration: 1050; easing.type: Easing.OutCubic }
            NumberAnimation { target: glow; property: "opacity"; from: 0; to: 0.10; duration: 1100 }
        }

        ParallelAnimation {
            NumberAnimation { target: titleText; property: "opacity"; from: 0; to: 1; duration: 420 }
            NumberAnimation { target: titleText; property: "anchors.topMargin"; from: 42; to: 26; duration: 420; easing.type: Easing.OutCubic }
        }
        NumberAnimation { target: subtitleText; property: "opacity"; from: 0; to: 1; duration: 380 }

        PauseAnimation { duration: 450 }

        ScriptAction { script: splash.finished() }
    }

    TapHandler {
        onTapped: {
            if (!introStarted)
                splash.beginIntro()
            else
                splash.finished()
        }
    }
}
