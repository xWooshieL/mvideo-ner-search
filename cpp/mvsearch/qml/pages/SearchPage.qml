// страница поиска: запрос -> факты чипами -> JSON по кнопке -> RecSys-ранжирование
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvSearch
import "../components"

Item {
    id: page

    property var facts: null
    property var ranked: []
    property bool jsonVisible: false

    // автозапуск запроса из окружения MV_QUERY (для скриншотов и демо)
    Component.onCompleted: {
        const q = SearchEngine.envQuery()
        if (q.length > 0) {
            searchField.text = q
            runSearch()
            if (Qt.application.arguments.indexOf("--json") >= 0)
                jsonVisible = true
        }
    }

    function runSearch() {
        const q = searchField.text.trim()
        if (q.length === 0)
            return
        facts = SearchEngine.extract(q)
        ranked = SearchEngine.rankCatalog(facts, 30)
        jsonArea.text = SearchEngine.extractJson(q)
        resultsAppear.restart()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 26
        spacing: 16

        PageHeader {
            title: qsTr("Умный поиск")
            subtitle: qsTr("Запрос превращается в структурированные факты, а карточки ранжируются только по ним")
        }

        // строка поиска
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            AppTextField {
                id: searchField
                Layout.fillWidth: true
                placeholderText: qsTr("Например: наушники logitech g pro x se 128гб …")
                font.pixelSize: Theme.fontMedium
                implicitHeight: 48
                onAccepted: page.runSearch()
            }

            AccentButton {
                text: qsTr("Найти")
                implicitHeight: 48
                onClicked: page.runSearch()
            }
        }

        // результаты
        Flickable {
            id: resultsFlick
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: resultsColumn.implicitHeight
            clip: true
            visible: page.facts !== null

            opacity: 0
            NumberAnimation {
                id: resultsAppear
                target: resultsFlick; property: "opacity"
                from: 0; to: 1; duration: 350; easing.type: Easing.OutCubic
            }

            ScrollBar.vertical: ScrollBar { }

            ColumnLayout {
                id: resultsColumn
                width: resultsFlick.width
                spacing: 14

                // --- извлечённые факты
                AppCard {
                    Layout.fillWidth: true
                    implicitHeight: factsColumn.implicitHeight + 36

                    ColumnLayout {
                        id: factsColumn
                        anchors.fill: parent
                        anchors.margins: 18
                        spacing: 12

                        RowLayout {
                            Layout.fillWidth: true

                            Text {
                                text: qsTr("Извлечённые факты")
                                font.pixelSize: Theme.fontMedium
                                font.family: Theme.fontFamily
                                font.weight: Font.Bold
                                color: Theme.text
                            }

                            Item { Layout.fillWidth: true }

                            Rectangle {
                                implicitWidth: latText.implicitWidth + 18
                                implicitHeight: 24
                                radius: 12
                                color: Theme.successSoft

                                Text {
                                    id: latText
                                    anchors.centerIn: parent
                                    text: page.facts
                                          ? qsTr("%1 мс").arg(Number(page.facts.latency_ms).toFixed(2))
                                          : ""
                                    font.pixelSize: 11
                                    font.family: Theme.fontFamily
                                    font.weight: Font.Bold
                                    color: Theme.success
                                }
                            }

                            GhostButton {
                                text: page.jsonVisible ? qsTr("Скрыть JSON") : qsTr("Показать JSON")
                                implicitHeight: 32
                                onClicked: page.jsonVisible = !page.jsonVisible
                            }
                        }

                        // чипы фактов
                        Flow {
                            Layout.fillWidth: true
                            spacing: 8

                            FactChip {
                                visible: page.facts && page.facts.brand.length > 0
                                tagRu: qsTr("бренд")
                                value: page.facts ? page.facts.brand : ""
                                tagColor: Theme.tagBrand
                            }

                            FactChip {
                                visible: page.facts && page.facts.category.length > 0
                                tagRu: qsTr("категория")
                                value: page.facts ? page.facts.category : ""
                                tagColor: Theme.tagCategory
                            }

                            FactChip {
                                visible: page.facts && page.facts.model.length > 0
                                tagRu: qsTr("модель")
                                value: page.facts ? page.facts.model : ""
                                tagColor: Theme.tagModel
                            }

                            Repeater {
                                model: page.facts ? Object.keys(page.facts.attributes) : []

                                FactChip {
                                    required property string modelData
                                    tagRu: modelData
                                    value: page.facts.attributes[modelData]
                                    tagColor: Theme.tagAttr
                                }
                            }

                            Text {
                                visible: page.facts
                                         && page.facts.brand.length === 0
                                         && page.facts.category.length === 0
                                         && page.facts.model.length === 0
                                         && Object.keys(page.facts.attributes).length === 0
                                text: qsTr("Фактов не найдено — попробуйте уточнить запрос")
                                font.pixelSize: Theme.fontBody
                                font.family: Theme.fontFamily
                                color: Theme.textTertiary
                            }
                        }

                        // раскрывающийся JSON
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: page.jsonVisible ? Math.min(jsonArea.implicitHeight + 24, 260) : 0
                            radius: Theme.radiusSmall
                            color: Theme.surfaceAlt
                            clip: true
                            visible: Layout.preferredHeight > 0

                            Behavior on Layout.preferredHeight {
                                NumberAnimation { duration: 260; easing.type: Easing.OutCubic }
                            }

                            Flickable {
                                anchors.fill: parent
                                anchors.margins: 12
                                contentHeight: jsonArea.implicitHeight
                                clip: true

                                ScrollBar.vertical: ScrollBar { }

                                TextEdit {
                                    id: jsonArea
                                    width: parent.width
                                    readOnly: true
                                    selectByMouse: true
                                    wrapMode: TextEdit.WrapAnywhere
                                    font.pixelSize: 11
                                    font.family: "Consolas"
                                    color: Theme.text
                                }
                            }
                        }
                    }
                }

                // --- RecSys
                RowLayout {
                    Layout.fillWidth: true
                    Layout.topMargin: 4

                    Text {
                        text: qsTr("Подбор по фактам")
                        font.pixelSize: Theme.fontMedium
                        font.family: Theme.fontFamily
                        font.weight: Font.Bold
                        color: Theme.text
                    }

                    Text {
                        text: qsTr("· %1 карточек ранжировано только по извлечённым фактам").arg(page.ranked.length)
                        font.pixelSize: Theme.fontSmall
                        font.family: Theme.fontFamily
                        color: Theme.textSecondary
                    }

                    Item { Layout.fillWidth: true }
                }

                Repeater {
                    model: page.ranked

                    AppCard {
                        id: itemCard
                        required property var modelData
                        required property int index

                        Layout.fillWidth: true
                        implicitHeight: 72

                        opacity: 0
                        Component.onCompleted: cardAppear.start()

                        SequentialAnimation {
                            id: cardAppear
                            PauseAnimation { duration: Math.min(itemCard.index * 35, 500) }
                            ParallelAnimation {
                                NumberAnimation { target: itemCard; property: "opacity"; from: 0; to: 1; duration: 300; easing.type: Easing.OutCubic }
                            }
                        }

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 16
                            anchors.rightMargin: 16
                            spacing: 14

                            // позиция
                            Rectangle {
                                Layout.preferredWidth: 34
                                Layout.preferredHeight: 34
                                radius: 17
                                color: itemCard.index === 0 ? Theme.accent : Theme.surfaceAlt

                                Text {
                                    anchors.centerIn: parent
                                    text: itemCard.index + 1
                                    font.pixelSize: 13
                                    font.family: Theme.fontFamily
                                    font.weight: Font.Bold
                                    color: itemCard.index === 0 ? "#ffffff" : Theme.textSecondary
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 3

                                Text {
                                    Layout.fillWidth: true
                                    text: itemCard.modelData.name
                                    font.pixelSize: Theme.fontBody
                                    font.family: Theme.fontFamily
                                    font.weight: Font.DemiBold
                                    color: Theme.text
                                    elide: Text.ElideRight
                                }

                                Text {
                                    text: qsTr("Совпало: %1").arg(itemCard.modelData.why)
                                    font.pixelSize: 11
                                    font.family: Theme.fontFamily
                                    color: Theme.textSecondary
                                    visible: itemCard.modelData.why.length > 0
                                }
                            }

                            // полоса релевантности
                            ColumnLayout {
                                Layout.preferredWidth: 110
                                spacing: 4

                                Text {
                                    text: qsTr("релевантность %1%").arg(Math.round(itemCard.modelData.match * 100))
                                    font.pixelSize: 10
                                    font.family: Theme.fontFamily
                                    color: Theme.textSecondary
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 6
                                    radius: 3
                                    color: Theme.surfaceAlt

                                    Rectangle {
                                        width: parent.width * itemCard.modelData.match
                                        height: parent.height
                                        radius: 3
                                        color: Theme.accent

                                        Behavior on width { NumberAnimation { duration: 420; easing.type: Easing.OutCubic } }
                                    }
                                }
                            }

                            Text {
                                text: itemCard.modelData.price > 0
                                      ? Number(itemCard.modelData.price).toLocaleString(Qt.locale("ru_RU"), "f", 0) + " ₽"
                                      : "—"
                                font.pixelSize: Theme.fontMedium
                                font.family: Theme.fontFamily
                                font.weight: Font.Bold
                                color: Theme.accent
                            }
                        }
                    }
                }
            }
        }

        // пустое состояние
        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: page.facts === null

            Column {
                anchors.centerIn: parent
                spacing: 12

                Image {
                    source: Theme.logoMarkSource
                    width: 72
                    height: 72
                    fillMode: Image.PreserveAspectFit
                    anchors.horizontalCenter: parent.horizontalCenter
                    opacity: 0.35
                }

                Text {
                    text: qsTr("Введите запрос, как в поиске магазина")
                    font.pixelSize: Theme.fontMedium
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                    anchors.horizontalCenter: parent.horizontalCenter
                }

                Text {
                    text: qsTr("Попробуйте: «айфон 15 про 256гб», «телевизор samsung 55 дюймов», «наушники logitech g pro»")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textTertiary
                    anchors.horizontalCenter: parent.horizontalCenter
                }
            }
        }
    }
}
