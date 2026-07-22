// настройки приложения разметки: сведения + удаление (как в ГК МОС)
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvLabel
import "components"

Item {
    id: root
    width: 0
    height: 0

    property alias settingsPopup: settingsPopup

    component OptionCheck : RowLayout {
        id: optRow
        property bool checked: false
        property string label
        signal toggled(bool value)
        spacing: 10

        Rectangle {
            width: 20
            height: 20
            radius: 5
            color: optRow.checked ? Theme.accent : "transparent"
            border.width: 1.5
            border.color: optRow.checked ? Theme.accent : Theme.borderStrong

            Behavior on color { ColorAnimation { duration: 120 } }

            Text {
                anchors.centerIn: parent
                text: "\u2713"
                visible: optRow.checked
                color: "#ffffff"
                font.pixelSize: 13
                font.bold: true
            }

            HoverHandler { cursorShape: Qt.PointingHandCursor }
            TapHandler { onTapped: optRow.toggled(!optRow.checked) }
        }

        Text {
            Layout.fillWidth: true
            text: optRow.label
            font.pixelSize: Theme.fontBody
            font.family: Theme.fontFamily
            color: Theme.text
            wrapMode: Text.WordWrap

            TapHandler { onTapped: optRow.toggled(!optRow.checked) }
        }
    }

    Popup {
        id: settingsPopup
        anchors.centerIn: Overlay.overlay
        width: 460
        modal: true
        focus: true
        padding: 0
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        background: Rectangle {
            radius: Theme.radiusLarge
            color: Theme.surface
            border.width: 1
            border.color: Theme.border
        }

        contentItem: Column {
            padding: 24
            spacing: 16

            Text {
                text: qsTr("Настройки")
                font.pixelSize: Theme.fontLarge
                font.family: Theme.fontFamily
                font.weight: Font.Bold
                color: Theme.text
            }

            Text {
                text: qsTr("М.Видео · Разметка · %1").arg(LabelStore.annotator)
                font.pixelSize: Theme.fontBody
                font.family: Theme.fontFamily
                color: Theme.textSecondary
                width: 390
                wrapMode: Text.WordWrap
            }

            Rectangle {
                width: 390
                height: 1
                color: Theme.border
            }

            Column {
                width: 390
                spacing: 4

                Text {
                    text: qsTr("Удалить приложение")
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: Theme.text
                }

                Text {
                    width: 390
                    wrapMode: Text.WordWrap
                    text: qsTr("Откроет деинсталлятор Windows. Можно сразу стереть свою разметку и настройки аннотатора.")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                }
            }

            Row {
                spacing: 8
                anchors.right: parent.right
                anchors.rightMargin: 24

                GhostButton {
                    text: qsTr("Закрыть")
                    implicitHeight: 38
                    onClicked: settingsPopup.close()
                }

                AccentButton {
                    text: qsTr("Удалить приложение")
                    danger: true
                    implicitHeight: 38
                    onClicked: {
                        settingsPopup.close()
                        uninstallOptionsPopup.open()
                    }
                }
            }
        }
    }

    Popup {
        id: uninstallOptionsPopup
        anchors.centerIn: Overlay.overlay
        width: 460
        modal: true
        focus: true
        padding: 0
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        property bool deleteLabels: true
        property bool deleteSettings: true

        background: Rectangle {
            radius: Theme.radiusLarge
            color: Theme.surface
            border.width: 1
            border.color: Theme.border
        }

        contentItem: Column {
            padding: 24
            spacing: 14

            Text {
                text: qsTr("Удалить приложение?")
                font.pixelSize: Theme.fontLarge
                font.family: Theme.fontFamily
                font.weight: Font.Bold
                color: Theme.text
            }

            Text {
                text: qsTr("Выбери, что удалить вместе с программой. После подтверждения приложение закроется и запустит деинсталлятор Windows.")
                font.pixelSize: Theme.fontBody
                font.family: Theme.fontFamily
                color: Theme.textSecondary
                wrapMode: Text.WordWrap
                width: 390
            }

            OptionCheck {
                width: 390
                label: qsTr("Удалить мою разметку (папка labels — BIO и match 1/0)")
                checked: uninstallOptionsPopup.deleteLabels
                onToggled: uninstallOptionsPopup.deleteLabels = value
            }

            OptionCheck {
                width: 390
                label: qsTr("Удалить настройки (выбор аннотатора и т.п.)")
                checked: uninstallOptionsPopup.deleteSettings
                onToggled: uninstallOptionsPopup.deleteSettings = value
            }

            Row {
                spacing: 8
                anchors.right: parent.right
                anchors.rightMargin: 24

                GhostButton {
                    text: qsTr("Отмена")
                    implicitHeight: 38
                    onClicked: uninstallOptionsPopup.close()
                }

                AccentButton {
                    text: qsTr("Удалить")
                    danger: true
                    implicitHeight: 38
                    onClicked: {
                        LabelStore.prepareForUninstall(
                            uninstallOptionsPopup.deleteLabels,
                            uninstallOptionsPopup.deleteSettings)
                        uninstallOptionsPopup.close()
                        LabelStore.launchUninstaller()
                    }
                }
            }
        }
    }
}
