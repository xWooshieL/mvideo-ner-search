// страница статистики: данные движка и метрики качества
import QtQuick
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
            title: qsTr("Статистика движка")
            subtitle: qsTr("Что загружено и как измеряли качество")
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 4
            columnSpacing: 12
            rowSpacing: 12

            StatTile {
                Layout.fillWidth: true
                label: qsTr("Брендов в словаре")
                value: SearchEngine.brandCount.toLocaleString(Qt.locale("ru_RU"), "f", 0)
            }

            StatTile {
                Layout.fillWidth: true
                label: qsTr("Категорий в словаре")
                value: SearchEngine.categoryCount.toLocaleString(Qt.locale("ru_RU"), "f", 0)
            }

            StatTile {
                Layout.fillWidth: true
                label: qsTr("Фраз-линеек (майнинг)")
                value: SearchEngine.modelPhraseCount.toLocaleString(Qt.locale("ru_RU"), "f", 0)
            }

            StatTile {
                Layout.fillWidth: true
                label: qsTr("Карточек каталога")
                value: SearchEngine.catalogCount.toLocaleString(Qt.locale("ru_RU"), "f", 0)
            }
        }

        AppCard {
            Layout.fillWidth: true
            implicitHeight: metricsCol.implicitHeight + 36

            ColumnLayout {
                id: metricsCol
                anchors.fill: parent
                anchors.margins: 18
                spacing: 10

                Text {
                    text: qsTr("Метрики качества (быстрый тест на выборке)")
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: Theme.text
                }

                Text {
                    text: qsTr("Числа получены проверкой поверх слабой разметки — честную оценку даст только ручная золотая разметка (в работе, 3×1500 запросов).")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }

                component MetricRow : RowLayout {
                    property string name
                    property real value

                    Layout.fillWidth: true
                    spacing: 12

                    Text {
                        text: name
                        font.pixelSize: Theme.fontBody
                        font.family: Theme.fontFamily
                        color: Theme.text
                        Layout.preferredWidth: 220
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 8
                        radius: 4
                        color: Theme.surfaceAlt

                        Rectangle {
                            width: parent.width * value
                            height: parent.height
                            radius: 4
                            color: Theme.accent
                        }
                    }

                    Text {
                        text: value.toFixed(2)
                        font.pixelSize: Theme.fontBody
                        font.family: Theme.fontFamily
                        font.weight: Font.Bold
                        color: Theme.accent
                        Layout.preferredWidth: 44
                        horizontalAlignment: Text.AlignRight
                    }
                }

                MetricRow { name: qsTr("Точность по токенам"); value: 0.91 }
                MetricRow { name: qsTr("F1 по сущностям"); value: 0.875 }
                MetricRow { name: qsTr("F1 бренды"); value: 0.95 }
                MetricRow { name: qsTr("F1 категории"); value: 0.82 }
                MetricRow { name: qsTr("F1 атрибуты"); value: 0.85 }
            }
        }

        AppCard {
            Layout.fillWidth: true
            implicitHeight: archCol.implicitHeight + 36

            ColumnLayout {
                id: archCol
                anchors.fill: parent
                anchors.margins: 18
                spacing: 8

                Text {
                    text: qsTr("Как устроен каскад")
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: Theme.text
                }

                Text {
                    Layout.fillWidth: true
                    text: qsTr("1. SpellFix: опечатки, гомоглифы (c/с), алиасы транслита (сони→sony), «16гь»→«16 гб».\n2. Словари: бренды, категории, линейки моделей из майнинга.\n3. Регулярки атрибутов: память, размер, мощность, связь и другие.\n4. Марковский типизатор: пара соседних токенов уточняет тип атрибута.\n5. RecSys: карточки каталога ранжируются только по извлечённым фактам.")
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
