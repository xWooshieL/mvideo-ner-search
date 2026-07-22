// разметка соответствия 1/0: запрос против карточки, гайд и авто-0 для пустых карточек
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Layouts
import MvLabel
import "components"

Item {
    id: page

    property int pos: LabelStore.firstUnlabeledMatch()

    readonly property var pair: LabelStore.pairs[pos] || ({})
    readonly property string skuName: (pair.sku_name || "").trim()
    readonly property bool emptyCard: skuName.length === 0
                                      || skuName.toLowerCase() === "nan"
                                      || skuName.toLowerCase() === "none"

    // пустая карточка -> авто-0
    onPosChanged: maybeAutoZero()
    Component.onCompleted: maybeAutoZero()

    function maybeAutoZero() {
        if (emptyCard && LabelStore.pairs.length > 0) {
            autoZeroTimer.restart()
        }
    }

    Timer {
        id: autoZeroTimer
        interval: 450
        onTriggered: {
            if (page.emptyCard)
                page.mark(0, true)
        }
    }

    function mark(label, autoLabel) {
        LabelStore.saveMatch(pos, pair, label, autoLabel === true)
        if (pos < LabelStore.pairs.length - 1)
            pos++
    }

    focus: true
    Keys.onPressed: (event) => {
        event.accepted = true
        if (event.key === Qt.Key_1) mark(1, false)
        else if (event.key === Qt.Key_0) mark(0, false)
        else if (event.key === Qt.Key_Left) { if (pos > 0) pos-- }
        else if (event.key === Qt.Key_Right) { if (pos < LabelStore.pairs.length - 1) pos++ }
        else event.accepted = false
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 10

        RowLayout {
            Layout.fillWidth: true

            Text {
                text: qsTr("Пара %1 из %2 · размечено %3")
                      .arg(page.pos + 1).arg(LabelStore.pairs.length).arg(LabelStore.matchDone)
                font.pixelSize: Theme.fontSmall
                font.family: Theme.fontFamily
                color: Theme.textSecondary
            }

            Item { Layout.fillWidth: true }

            GhostButton {
                text: qsTr("Открыть папку с разметкой")
                implicitHeight: 32
                onClicked: LabelStore.openLabelsFolder()
            }
        }

        NiceProgress {
            Layout.fillWidth: true
            done: LabelStore.matchDone
            total: LabelStore.pairs.length
        }

        // гайд: когда 1, когда 0
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: guideCol.implicitHeight + 22
            radius: Theme.radiusMedium
            color: Theme.accentSoft

            ColumnLayout {
                id: guideCol
                anchors.fill: parent
                anchors.margins: 12
                spacing: 5

                Text {
                    Layout.fillWidth: true
                    text: qsTr("Когда ставить 1: карточка — именно тот товар (или его вариант по цвету/памяти), который искали. «ноутбук asus» + ASUS VivoBook — это 1.")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    font.weight: Font.DemiBold
                    color: Theme.success
                    wrapMode: Text.WordWrap
                }

                Text {
                    Layout.fillWidth: true
                    text: qsTr("Когда ставить 0: другой тип товара, аксессуар вместо товара, другой бренд, явно случайный клик. «айфон 15» + чехол для айфона — это 0. Если карточки нет вовсе — 0 ставится автоматически.")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    font.weight: Font.DemiBold
                    color: Theme.accent
                    wrapMode: Text.WordWrap
                }
            }
        }

        // пара запрос-карточка
        AppCard {
            Layout.fillWidth: true
            implicitHeight: pairCol.implicitHeight + 32

            ColumnLayout {
                id: pairCol
                anchors.fill: parent
                anchors.margins: 16
                spacing: 6

                Text {
                    text: qsTr("Запрос пользователя:")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                }

                Text {
                    Layout.fillWidth: true
                    text: "«" + (page.pair.query || "") + "»"
                    font.pixelSize: 17
                    font.family: Theme.fontFamily
                    font.weight: Font.Bold
                    color: Theme.accent
                    wrapMode: Text.WordWrap
                }

                Text {
                    Layout.topMargin: 6
                    text: qsTr("Карточка товара:")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                }

                Text {
                    Layout.fillWidth: true
                    text: page.emptyCard ? qsTr("— карточки нет (пустой клик), авто-разметка: 0 —") : page.skuName
                    font.pixelSize: Theme.fontBody
                    font.family: Theme.fontFamily
                    font.weight: Font.DemiBold
                    color: page.emptyCard ? Theme.textTertiary : Theme.text
                    wrapMode: Text.WordWrap
                }

                Text {
                    visible: !page.emptyCard
                    text: qsTr("Бренд: %1 · Цена: %2")
                          .arg(page.pair.brand || "—")
                          .arg(page.pair.price > 0
                               ? Number(page.pair.price).toLocaleString(Qt.locale("ru_RU"), "f", 0) + " ₽"
                               : "—")
                    font.pixelSize: Theme.fontSmall
                    font.family: Theme.fontFamily
                    color: Theme.textSecondary
                }
            }
        }

        // кнопки 1/0
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            AccentButton {
                Layout.fillWidth: true
                implicitHeight: 52
                enabled: !page.emptyCard
                text: qsTr("1 — соответствует")
                onClicked: page.mark(1, false)

                background: Rectangle {
                    radius: Theme.radiusMedium
                    color: parent.enabled
                           ? (parent.hovered ? Qt.darker(Theme.success, 1.12) : Theme.success)
                           : Theme.surfaceAlt

                    Behavior on color { ColorAnimation { duration: 150 } }
                }
            }

            AccentButton {
                Layout.fillWidth: true
                implicitHeight: 52
                enabled: !page.emptyCard
                text: qsTr("0 — не соответствует")
                onClicked: page.mark(0, false)
            }
        }

        RowLayout {
            Layout.fillWidth: true

            GhostButton {
                text: qsTr("← Назад")
                onClicked: if (page.pos > 0) page.pos--
            }

            Item { Layout.fillWidth: true }

            GhostButton {
                text: qsTr("Пропустить →")
                onClicked: if (page.pos < LabelStore.pairs.length - 1) page.pos++
            }
        }

        // история
        Text {
            text: qsTr("История (клик — открыть пару):")
            font.pixelSize: Theme.fontSmall
            font.family: Theme.fontFamily
            color: Theme.textSecondary
        }

        AppCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Theme.surfaceAlt

            ListView {
                id: histList
                anchors.fill: parent
                anchors.margins: 10
                clip: true
                spacing: 3
                model: LabelStore.matchDone >= 0 ? LabelStore.matchHistory(200) : []

                ScrollBar.vertical: ScrollBar { }

                delegate: Rectangle {
                    required property var modelData
                    width: histList.width
                    height: 24
                    radius: 4
                    color: rowHover.hovered ? Theme.hover : "transparent"

                    Row {
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.leftMargin: 6
                        spacing: 8

                        Rectangle {
                            width: 34
                            height: 16
                            radius: 4
                            color: modelData.label === 1 ? Theme.successSoft : Theme.errorSoft
                            anchors.verticalCenter: parent.verticalCenter

                            Text {
                                anchors.centerIn: parent
                                text: modelData.label + (modelData.auto ? " а" : "")
                                font.pixelSize: 9
                                font.family: Theme.fontFamily
                                font.weight: Font.Bold
                                color: modelData.label === 1 ? Theme.success : Theme.error
                            }
                        }

                        Text {
                            text: "#" + (modelData.index + 1) + " · " + modelData.query
                                  + " ↔ " + (modelData.sku_name || "—").substring(0, 60)
                            font.pixelSize: 10
                            font.family: Theme.fontFamily
                            color: Theme.textSecondary
                            anchors.verticalCenter: parent.verticalCenter
                        }
                    }

                    HoverHandler { id: rowHover }
                    TapHandler { onTapped: page.pos = modelData.index }
                }
            }
        }

        Text {
            text: qsTr("Клавиши: 1 / 0 — метка · ←/→ — навигация")
            font.pixelSize: 10
            font.family: Theme.fontFamily
            color: Theme.textTertiary
        }
    }
}
