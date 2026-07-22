// настройки: тема и информация о приложении
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvSearch
import "../components"

Item {
    id: page

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
        id: uninstallOptionsPopup
        anchors.centerIn: Overlay.overlay
        width: 440
        modal: true
        focus: true
        padding: 0
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

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
                text: qsTr("Программа закроется и запустит деинсталлятор Windows. После подтверждения действие не отменить.")
                font.pixelSize: Theme.fontBody
                font.family: Theme.fontFamily
                color: Theme.textSecondary
                wrapMode: Text.WordWrap
                width: 390
            }

            OptionCheck {
                width: 390
                label: qsTr("Удалить настройки приложения (тема, сохранённые параметры)")
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
                        SearchEngine.prepareForUninstall(uninstallOptionsPopup.deleteSettings)
                        uninstallOptionsPopup.close()
                        SearchEngine.launchUninstaller()
                    }
                }
            }
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 26
        spacing: 16

        PageHeader {
            title: qsTr("Настройки")
            subtitle: qsTr("Оформление и сведения о приложении")
        }

        AppCard {
            Layout.fillWidth: true
            implicitHeight: themeRow.implicitHeight + 36

            RowLayout {
                id: themeRow
                anchors.fill: parent
                anchors.margins: 18
                spacing: 14

                Column {
                    Layout.fillWidth: true
                    spacing: 3

                    Text {
                        text: qsTr("Тёмная тема")
                        font.pixelSize: Theme.fontBody
                        font.family: Theme.fontFamily
                        font.weight: Font.DemiBold
                        color: Theme.text
                    }

                    Text {
                        text: qsTr("Красно-чёрное оформление вместо красно-белого")
                        font.pixelSize: Theme.fontSmall
                        font.family: Theme.fontFamily
                        color: Theme.textSecondary
                    }
                }

                Switch {
                    id: darkSwitch
                    checked: Theme.dark
                    onToggled: Theme.dark = checked

                    indicator: Rectangle {
                        implicitWidth: 46
                        implicitHeight: 26
                        radius: 13
                        color: darkSwitch.checked ? Theme.accent : Theme.surfaceAlt
                        border.width: 1
                        border.color: darkSwitch.checked ? Theme.accent : Theme.border

                        Behavior on color { ColorAnimation { duration: 180 } }

                        Rectangle {
                            x: darkSwitch.checked ? parent.width - width - 3 : 3
                            anchors.verticalCenter: parent.verticalCenter
                            width: 20
                            height: 20
                            radius: 10
                            color: "#ffffff"

                            Behavior on x { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }
                        }
                    }
                }
            }
        }

        AppCard {
            Layout.fillWidth: true
            implicitHeight: aboutCol.implicitHeight + 36

            ColumnLayout {
                id: aboutCol
                anchors.fill: parent
                anchors.margins: 18
                spacing: 8

                Text {
                    text: qsTr("О приложении")
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: Theme.text
                }

                Text {
                    Layout.fillWidth: true
                    text: qsTr("М.Видео · Умный поиск — MVP извлечения фактов из поисковых запросов.\nДвижок на C++: словари, регулярки, марковский типизатор, ранжирование по фактам.\nВерсия %1 · буткемп-команда: Никита, Некит, Лиза").arg(SearchEngine.appVersion)
                    font.pixelSize: Theme.fontBody
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                    wrapMode: Text.WordWrap
                    lineHeight: 1.35
                }
            }
        }

        AppCard {
            Layout.fillWidth: true
            implicitHeight: dangerRow.implicitHeight + 36
            border.color: Theme.errorSoft

            RowLayout {
                id: dangerRow
                anchors.fill: parent
                anchors.margins: 18
                spacing: 12

                Column {
                    Layout.fillWidth: true
                    spacing: 3

                    Text {
                        text: qsTr("Удалить приложение")
                        font.pixelSize: Theme.fontMedium
                        font.family: Theme.fontFamily
                        font.weight: Font.Bold
                        color: Theme.text
                    }

                    Text {
                        width: parent.width
                        wrapMode: Text.WordWrap
                        text: qsTr("Откроет деинсталлятор Windows. Можно сразу очистить настройки приложения.")
                        font.pixelSize: Theme.fontSmall
                        font.family: Theme.fontFamily
                        color: Theme.textSecondary
                    }
                }

                AccentButton {
                    text: qsTr("Удалить")
                    danger: true
                    onClicked: uninstallOptionsPopup.open()
                }
            }
        }

        Item { Layout.fillHeight: true }
    }
}
