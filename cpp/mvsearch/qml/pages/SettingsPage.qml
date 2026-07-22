// настройки: тема и информация о приложении
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvSearch
import "../components"

Item {
    id: page

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

        Item { Layout.fillHeight: true }
    }
}
