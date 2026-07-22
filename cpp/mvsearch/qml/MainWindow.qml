// главное окно: сайдбар + страницы поиска, статистики, настроек
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvSearch
import "components"
import "pages"

Item {
    id: mainWindow

    property int currentPage: Qt.application.arguments.indexOf("--stats") >= 0 ? 1 : 0

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ===== сайдбар =====
        Rectangle {
            id: sidebar
            Layout.preferredWidth: 236
            Layout.fillHeight: true
            color: Theme.surface

            Rectangle {
                anchors.right: parent.right
                width: 1
                height: parent.height
                color: Theme.border
            }

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 14
                spacing: 4

                // логотип + название
                Item {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 52

                    Row {
                        anchors.left: parent.left
                        anchors.leftMargin: 6
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: 10

                        Image {
                            source: Theme.logoMarkSource
                            width: 34
                            height: 34
                            fillMode: Image.PreserveAspectFit
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        Column {
                            anchors.verticalCenter: parent.verticalCenter
                            spacing: 0

                            Text {
                                text: qsTr("М.Видео")
                                font.pixelSize: 16
                                font.family: Theme.fontFamily
                                font.weight: Font.Bold
                                color: Theme.text
                            }
                            Text {
                                text: qsTr("Умный поиск")
                                font.pixelSize: 11
                                font.family: Theme.fontFamily
                                color: Theme.textSecondary
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border; Layout.bottomMargin: 8 }

                SideBarItem {
                    Layout.fillWidth: true
                    label: qsTr("Поиск")
                    icon: "\uE721"
                    active: currentPage === 0
                    onClicked: currentPage = 0
                }

                SideBarItem {
                    Layout.fillWidth: true
                    label: qsTr("Статистика")
                    icon: "\uE9D9"
                    active: currentPage === 1
                    onClicked: currentPage = 1
                }

                SideBarItem {
                    Layout.fillWidth: true
                    label: qsTr("Настройки")
                    icon: "\uE713"
                    active: currentPage === 2
                    onClicked: currentPage = 2
                }

                Item { Layout.fillHeight: true }

                // статус движка внизу сайдбара
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 58
                    radius: Theme.radiusSmall
                    color: Theme.accentSoft

                    Column {
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.leftMargin: 12
                        spacing: 3

                        Row {
                            spacing: 6

                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: SearchEngine.ready ? Theme.success : Theme.warning
                                anchors.verticalCenter: parent.verticalCenter
                            }

                            Text {
                                text: SearchEngine.ready ? qsTr("Движок готов") : qsTr("Нет данных")
                                font.pixelSize: Theme.fontSmall
                                font.family: Theme.fontFamily
                                font.weight: Font.DemiBold
                                color: Theme.text
                            }
                        }

                        Text {
                            text: qsTr("%1 брендов · %2 категорий")
                                  .arg(SearchEngine.brandCount).arg(SearchEngine.categoryCount)
                            font.pixelSize: 10
                            font.family: Theme.fontFamily
                            color: Theme.textSecondary
                        }
                    }
                }
            }
        }

        // ===== контент =====
        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: currentPage

            SearchPage { }
            StatsPage { }
            SettingsPage { }
        }
    }
}
