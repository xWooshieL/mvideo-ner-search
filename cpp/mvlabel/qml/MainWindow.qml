// главное окно разметки: красная шапка с переключателем режимов
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvLabel
import "components"

Item {
    id: mainWindow

    property int mode: LabelStore.envValue("MV_DEMO") === "match" ? 1 : 0   // 0 = BIO, 1 = match

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // красная шапка
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 58
            color: Theme.accent

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 22
                anchors.rightMargin: 22
                spacing: 12

                Image {
                    source: "qrc:/qt/qml/MvLabel/assets/logo_mark_white.png"
                    Layout.preferredWidth: 26
                    Layout.preferredHeight: 26
                    fillMode: Image.PreserveAspectFit
                }

                Text {
                    text: qsTr("М.Видео · Разметка · %1").arg(LabelStore.annotator)
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: "#ffffff"
                }

                Item { Layout.fillWidth: true }

                component ModeButton : Rectangle {
                    property string label
                    property bool active: false
                    signal clicked()

                    implicitWidth: modeText.implicitWidth + 30
                    implicitHeight: 34
                    radius: 8
                    color: active ? "#ffffff" : Qt.rgba(1, 1, 1, 0.18)

                    Behavior on color { ColorAnimation { duration: 160 } }

                    Text {
                        id: modeText
                        anchors.centerIn: parent
                        text: parent.label
                        font.pixelSize: Theme.fontBody
                        font.family: Theme.fontFamily
                        font.weight: parent.active ? Font.Bold : Font.Medium
                        color: parent.active ? Theme.accent : "#ffffff"
                    }

                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: parent.clicked() }
                }

                ModeButton {
                    label: qsTr("BIO-разметка")
                    active: mainWindow.mode === 0
                    onClicked: mainWindow.mode = 0
                }

                ModeButton {
                    label: qsTr("Соответствие 1/0")
                    active: mainWindow.mode === 1
                    onClicked: mainWindow.mode = 1
                }
            }
        }

        // контент
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: mainWindow.mode

            BioWizard {
                id: bioWizard
                focus: mainWindow.mode === 0
            }

            MatchPage {
                id: matchPage
                focus: mainWindow.mode === 1
            }
        }
    }

    onModeChanged: {
        if (mode === 0)
            bioWizard.forceActiveFocus()
        else
            matchPage.forceActiveFocus()
    }
}
