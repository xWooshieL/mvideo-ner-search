// трёхэтапный мастер BIO-разметки: B/I/O -> тип сущности -> подтип атрибута
// фон (история) заблюрен, запрос разбит на блоки по центру
import QtQuick
import QtQuick.Controls.Basic
import QtQuick.Effects
import QtQuick.Layouts
import MvLabel
import "components"

Item {
    id: wizard

    // ---- состояние
    property int pos: LabelStore.firstUnlabeledBio()
    property int stage: 0            // 0 = BIO, 1 = тип, 2 = подтип
    property int cursor: 0           // активный блок (этап 0/1) или атрибут (этап 2)
    property int subChoice: 0
    property var tokens: []
    property var bio: []             // "B"/"I"/"O"/""
    property var cats: []            // тип для каждого токена
    property var subs: []            // подтип для ATTR
    property var entityIdx: []       // индексы токенов с B/I (этап 1)
    property var attrIdx: []         // индексы токенов ATTR (этап 2)

    readonly property var categories: ["BRAND", "MODEL", "CATEGORY", "ATTR", "GENRE"]
    readonly property var catColors: [Theme.tagBrand, Theme.tagModel, Theme.tagCategory, Theme.tagAttr, Theme.tagGenre]
    readonly property var catShort: [
        qsTr("производитель: apple, dyson, «самсунг»"),
        qsTr("линейка/модель: zenbook, v15, galaxy s24"),
        qsTr("тип товара: ноутбук, пылесос"),
        qsTr("характеристика: 16 гб, красный, wi-fi"),
        qsTr("жанр для игр/книг/фильмов")
    ]
    readonly property var catFull: [
        qsTr("Производитель товара. Ставим, когда слово — название компании-бренда, в том числе русской транслитерацией: «самсунг», «дайсон»."),
        qsTr("Линейка или модель после бренда: zenbook, v15, galaxy s24, airpods pro. Уточняет конкретную серию устройства."),
        qsTr("Тип товара: ноутбук, телефон, пылесос. Слово, которое говорит, ЧТО ищет человек."),
        qsTr("Характеристика: числа с единицами, цвета, свойства. Подтип уточним на следующем шаге."),
        qsTr("Жанр или тематика для игр, книг, фильмов: хоррор, стратегия. Ставим редко — только для медиа-товаров.")
    ]

    readonly property var subtypes: [
        { code: "memory_storage", ru: qsTr("память / накопитель"), desc: qsTr("объём памяти: 16 гб, 512 gb, 1 тб") },
        { code: "size", ru: qsTr("размер / диагональ"), desc: qsTr("габариты и диагонали: 55 дюймов, 60 см") },
        { code: "color", ru: qsTr("цвет"), desc: qsTr("белый, чёрный, красный…") },
        { code: "connectivity", ru: qsTr("связь"), desc: qsTr("wi-fi, bluetooth, nfc, 5g, usb-c") },
        { code: "weight", ru: qsTr("вес"), desc: qsTr("2 кг, 500 г") },
        { code: "volume", ru: qsTr("объём"), desc: qsTr("1 л, 500 мл") },
        { code: "power", ru: qsTr("мощность"), desc: qsTr("2000 вт, 1.5 квт") },
        { code: "resolution", ru: qsTr("разрешение"), desc: qsTr("4k, full hd, 1920x1080") },
        { code: "other", ru: qsTr("другое (ввести вручную)"), desc: qsTr("если ни один подтип не подходит") }
    ]

    function loadQuery() {
        const q = LabelStore.queries[pos] || ""
        tokens = q.split(/\s+/).filter(t => t.length > 0)
        const saved = LabelStore.bioRecord(pos)
        let b = [], c = [], s = []
        for (let i = 0; i < tokens.length; ++i) {
            b.push(""); c.push(""); s.push("")
            if (saved && saved.tags && i < saved.tags.length) {
                const tg = saved.tags[i]
                if (tg === "O") { b[i] = "O" }
                else if (tg.indexOf("-") > 0) {
                    b[i] = tg.split("-")[0]
                    c[i] = tg.split("-")[1]
                }
                if (saved.subtypes && saved.subtypes[String(i)])
                    s[i] = saved.subtypes[String(i)]
            }
        }
        bio = b; cats = c; subs = s
        stage = 0; cursor = 0; subChoice = 0
        manualField.visible = false
    }

    Component.onCompleted: {
        loadQuery()
        applyDemo()
    }

    // демо-настройка для скриншотов: MV_DEMO = stage1 | stage2 | stage3
    function applyDemo() {
        const demo = LabelStore.envValue("MV_DEMO")
        if (demo === "" || tokens.length === 0)
            return
        const pat = ["B", "I", "O", "B", "I"]
        let b = [], c = []
        for (let i = 0; i < tokens.length; ++i) {
            b.push(pat[i % pat.length])
            c.push("")
        }
        if (demo === "stage1") {
            b[tokens.length - 1] = ""
            bio = b
            cursor = Math.min(2, tokens.length - 1)
            return
        }
        const catPat = ["CATEGORY", "ATTR", "", "BRAND", "MODEL"]
        entityIdx = []
        for (let i = 0; i < tokens.length; ++i) {
            if (b[i] === "B" || b[i] === "I") {
                entityIdx.push(i)
                c[i] = catPat[i % catPat.length] || "ATTR"
            }
        }
        bio = b; cats = c
        if (demo === "stage2") {
            stage = 1
            cursor = 0
        } else if (demo === "stage3") {
            attrIdx = []
            for (let i = 0; i < tokens.length; ++i)
                if (c[i] === "ATTR")
                    attrIdx.push(i)
            if (attrIdx.length === 0) {
                c[entityIdx[0]] = "ATTR"
                cats = c
                attrIdx = [entityIdx[0]]
            }
            stage = 2
            cursor = 0
            subChoice = 0
        }
    }

    // ---- этап 0: BIO
    function setBio(tag) {
        let b = bio.slice()
        b[cursor] = tag
        if (tag === "O") {
            let c = cats.slice(); c[cursor] = ""; cats = c
            let s = subs.slice(); s[cursor] = ""; subs = s
        }
        bio = b
        if (cursor < tokens.length - 1)
            cursor++
    }

    function bioBackspace() {
        let b = bio.slice()
        if (b[cursor] !== "") {
            b[cursor] = ""
            let c = cats.slice(); c[cursor] = ""; cats = c
            bio = b
        } else if (cursor > 0) {
            cursor--
        }
    }

    function bioEnter() {
        const unset = tokens.filter((t, i) => bio[i] === "")
        if (unset.length > 0) {
            infoDialog.text = qsTr("Остались блоки без метки: %1").arg(unset.join(", "))
            infoDialog.open()
            return
        }
        confirmDialog.text = qsTr("Вы подтверждаете разметку B/I/O?")
        confirmDialog.acceptAction = function() {
            // дефолтный тип для сущностей
            let c = cats.slice()
            entityIdx = []
            for (let i = 0; i < tokens.length; ++i) {
                if (bio[i] === "B" || bio[i] === "I") {
                    entityIdx.push(i)
                    if (c[i] === "") c[i] = "BRAND"
                }
            }
            cats = c
            if (entityIdx.length === 0) { save(); return }
            stage = 1
            cursor = 0
        }
        confirmDialog.open()
    }

    // ---- этап 1: типы
    function setType(catIndex) {
        let c = cats.slice()
        c[entityIdx[cursor]] = categories[catIndex]
        cats = c
        if (cursor < entityIdx.length - 1)
            cursor++
    }

    function cycleType(d) {
        let c = cats.slice()
        const i = entityIdx[cursor]
        let ci = categories.indexOf(c[i] || "BRAND")
        ci = (ci + d + categories.length) % categories.length
        c[i] = categories[ci]
        cats = c
    }

    function typeEnter() {
        if (cursor < entityIdx.length - 1) {
            cursor++
            return
        }
        confirmDialog.text = qsTr("Точно проставили все типы?")
        confirmDialog.acceptAction = function() {
            attrIdx = []
            for (let i = 0; i < tokens.length; ++i)
                if ((bio[i] === "B" || bio[i] === "I") && cats[i] === "ATTR")
                    attrIdx.push(i)
            if (attrIdx.length === 0) { save(); return }
            stage = 2
            cursor = 0
            subChoice = 0
        }
        confirmDialog.open()
    }

    // ---- этап 2: подтипы
    function subEnter() {
        const st = subtypes[subChoice]
        if (st.code === "other") {
            manualField.visible = true
            manualField.forceActiveFocus()
            return
        }
        let s = subs.slice()
        s[attrIdx[cursor]] = st.code
        subs = s
        subNext()
    }

    function manualDone() {
        const txt = manualField.text.trim()
        if (txt.length === 0)
            return
        let s = subs.slice()
        s[attrIdx[cursor]] = txt
        subs = s
        manualField.text = ""
        manualField.visible = false
        wizard.forceActiveFocus()
        subNext()
    }

    function subNext() {
        if (cursor < attrIdx.length - 1) {
            cursor++
            subChoice = 0
        } else {
            save()
        }
    }

    // ---- сохранение и переходы
    function save() {
        let tags = []
        let st = ({})
        for (let i = 0; i < tokens.length; ++i) {
            tags.push(bio[i] === "O" || bio[i] === "" ? "O" : bio[i] + "-" + cats[i])
            if (subs[i] !== "")
                st[String(i)] = subs[i]
        }
        LabelStore.saveBio(pos, tokens.join(" "), tags, st)
        if (pos < LabelStore.queries.length - 1)
            pos++
        loadQuery()
    }

    function prevQuery() {
        if (pos > 0) { pos--; loadQuery() }
    }

    function nextQuery() {
        if (pos < LabelStore.queries.length - 1) { pos++; loadQuery() }
    }

    // ---- клавиатура
    focus: true
    Keys.onPressed: (event) => {
        if (manualField.visible) {
            if (event.key === Qt.Key_Escape) {
                manualField.visible = false
                event.accepted = true
            }
            return
        }
        event.accepted = true
        if (stage === 0) {
            if (event.key === Qt.Key_B) setBio("B")
            else if (event.key === Qt.Key_I) setBio("I")
            else if (event.key === Qt.Key_O) setBio("O")
            else if (event.key === Qt.Key_Left) cursor = Math.max(0, cursor - 1)
            else if (event.key === Qt.Key_Right) cursor = Math.min(tokens.length - 1, cursor + 1)
            else if (event.key === Qt.Key_Backspace) bioBackspace()
            else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) bioEnter()
            else event.accepted = false
        } else if (stage === 1) {
            if (event.key >= Qt.Key_1 && event.key <= Qt.Key_5) setType(event.key - Qt.Key_1)
            else if (event.key === Qt.Key_Up) cycleType(-1)
            else if (event.key === Qt.Key_Down) cycleType(1)
            else if (event.key === Qt.Key_Left) cursor = Math.max(0, cursor - 1)
            else if (event.key === Qt.Key_Right) cursor = Math.min(entityIdx.length - 1, cursor + 1)
            else if (event.key === Qt.Key_Backspace) { stage = 0; cursor = tokens.length - 1 }
            else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) typeEnter()
            else event.accepted = false
        } else {
            if (event.key >= Qt.Key_1 && event.key <= Qt.Key_9) {
                const idx = event.key - Qt.Key_1
                if (idx < subtypes.length) { subChoice = idx; subEnter() }
            }
            else if (event.key === Qt.Key_Up) subChoice = (subChoice - 1 + subtypes.length) % subtypes.length
            else if (event.key === Qt.Key_Down) subChoice = (subChoice + 1) % subtypes.length
            else if (event.key === Qt.Key_Backspace) { stage = 1; cursor = 0 }
            else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) subEnter()
            else event.accepted = false
        }
    }

    function catColor(cat) {
        const i = categories.indexOf(cat)
        return i >= 0 ? catColors[i] : Theme.tagO
    }

    // ================= вёрстка =================
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 10

        RowLayout {
            Layout.fillWidth: true

            Text {
                text: qsTr("Размечено %1 из %2").arg(LabelStore.bioDone).arg(LabelStore.queries.length)
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
            done: LabelStore.bioDone
            total: LabelStore.queries.length
        }

        // ---- сцена: блюр-фон (история) + мастер поверх
        Item {
            id: scene
            Layout.fillWidth: true
            Layout.fillHeight: true

            // фоновая история — блюрим
            AppCard {
                id: backdrop
                anchors.fill: parent
                color: Theme.surfaceAlt

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 8

                    Text {
                        text: qsTr("История разметок (клик — открыть запрос)")
                        font.pixelSize: 11
                        font.family: Theme.fontFamily
                        font.weight: Font.Bold
                        color: Theme.textSecondary
                    }

                    ListView {
                        id: historyList
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 4
                        model: LabelStore.bioDone >= 0 ? LabelStore.bioHistory(200) : []

                        delegate: Rectangle {
                            required property var modelData
                            width: historyList.width
                            height: 26
                            radius: 4
                            color: "transparent"

                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                anchors.left: parent.left
                                anchors.leftMargin: 8
                                width: parent.width - 16
                                text: "#" + (modelData.index + 1) + " · " + modelData.query
                                      + "  →  " + modelData.tags.join(" ")
                                font.pixelSize: 10
                                font.family: Theme.fontFamily
                                color: Theme.textSecondary
                                elide: Text.ElideRight
                            }

                            TapHandler {
                                onTapped: {
                                    wizard.pos = modelData.index
                                    wizard.loadQuery()
                                }
                            }
                        }
                    }
                }

                layer.enabled: true
                layer.effect: MultiEffect {
                    blurEnabled: true
                    blur: 0.85
                    blurMax: 24
                }
            }

            // передний план — мастер
            Rectangle {
                id: front
                anchors.centerIn: parent
                width: Math.min(parent.width - 40, 920)
                height: Math.min(parent.height - 24, frontCol.implicitHeight + 44)
                radius: Theme.radiusLarge
                color: Qt.alpha(Theme.surface, 0.96)
                border.width: 1
                border.color: Theme.border

                ColumnLayout {
                    id: frontCol
                    anchors.fill: parent
                    anchors.margins: 22
                    spacing: 12

                    // заголовок этапа
                    Text {
                        Layout.alignment: Qt.AlignHCenter
                        text: wizard.stage === 0 ? qsTr("ЭТАП 1 · РАЗМЕТКА B / I / O")
                              : wizard.stage === 1 ? qsTr("ЭТАП 2 · ТИП СУЩНОСТИ (1–5)")
                              : qsTr("ЭТАП 3 · ПОДТИП АТРИБУТА")
                        font.pixelSize: Theme.fontSmall
                        font.family: Theme.fontFamily
                        font.weight: Font.Bold
                        font.letterSpacing: 1.4
                        color: Theme.accent
                    }

                    Text {
                        Layout.alignment: Qt.AlignHCenter
                        text: qsTr("Запрос %1 из %2").arg(wizard.pos + 1).arg(LabelStore.queries.length)
                        font.pixelSize: Theme.fontSmall
                        font.family: Theme.fontFamily
                        color: Theme.textSecondary
                    }

                    // блоки токенов по центру
                    Flow {
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignHCenter
                        spacing: 10

                        Repeater {
                            model: wizard.tokens

                            Column {
                                id: blockCol
                                required property string modelData
                                required property int index

                                readonly property bool isActive: {
                                    if (wizard.stage === 0) return index === wizard.cursor
                                    if (wizard.stage === 1) return wizard.entityIdx[wizard.cursor] === index
                                    return wizard.attrIdx[wizard.cursor] === index
                                }
                                readonly property string myBio: wizard.bio[index] || ""
                                readonly property string myCat: wizard.cats[index] || ""
                                readonly property string mySub: wizard.subs[index] || ""

                                spacing: 4

                                // метка над блоком
                                Rectangle {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    width: tagText.implicitWidth + 14
                                    height: 20
                                    radius: 5
                                    visible: blockCol.myBio !== ""
                                    color: blockCol.myBio === "O" ? Theme.tagO
                                           : blockCol.myCat !== "" ? wizard.catColor(blockCol.myCat)
                                           : Theme.tagBrand

                                    Text {
                                        id: tagText
                                        anchors.centerIn: parent
                                        text: {
                                            if (blockCol.myBio === "O") return "O"
                                            let t = blockCol.myBio
                                            if (blockCol.myCat !== "") t += "-" + blockCol.myCat
                                            if (blockCol.mySub !== "") t += " · " + blockCol.mySub
                                            return t
                                        }
                                        font.pixelSize: 9
                                        font.family: Theme.fontFamily
                                        font.weight: Font.Bold
                                        color: "#ffffff"
                                    }
                                }

                                Item {
                                    height: 20
                                    width: 1
                                    visible: blockCol.myBio === ""
                                }

                                // сам блок
                                Rectangle {
                                    width: tokenText.implicitWidth + 34
                                    height: 50
                                    radius: 10
                                    color: blockCol.myBio === "O" ? Theme.surfaceAlt
                                           : blockCol.myBio !== "" && blockCol.myCat !== "" ? wizard.catColor(blockCol.myCat)
                                           : blockCol.myBio !== "" ? Theme.tagBrand
                                           : Theme.surface
                                    border.width: blockCol.isActive ? 3 : 1
                                    border.color: blockCol.isActive ? Theme.accent : Theme.border

                                    Behavior on color { ColorAnimation { duration: 180 } }

                                    Text {
                                        id: tokenText
                                        anchors.centerIn: parent
                                        text: blockCol.modelData
                                        font.pixelSize: Theme.fontMedium
                                        font.family: Theme.fontFamily
                                        font.weight: Font.Bold
                                        color: blockCol.myBio === "O" ? Theme.textSecondary
                                               : blockCol.myBio !== "" ? "#ffffff"
                                               : Theme.text
                                    }

                                    TapHandler {
                                        onTapped: {
                                            if (wizard.stage === 0)
                                                wizard.cursor = blockCol.index
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // панель подсказки этапа
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: panelCol.implicitHeight + 22
                        radius: Theme.radiusMedium
                        color: Theme.accentSoft

                        ColumnLayout {
                            id: panelCol
                            anchors.fill: parent
                            anchors.margins: 12
                            spacing: 4

                            Text {
                                Layout.fillWidth: true
                                text: {
                                    if (wizard.stage === 0)
                                        return qsTr("Клавиши: B — начало сущности · I — продолжение · O — не сущность")
                                    if (wizard.stage === 1)
                                        return qsTr("Блок «%1» — выберите тип:").arg(wizard.tokens[wizard.entityIdx[wizard.cursor]] || "")
                                    return qsTr("Атрибут «%1» — выберите подтип:").arg(wizard.tokens[wizard.attrIdx[wizard.cursor]] || "")
                                }
                                font.pixelSize: Theme.fontSmall
                                font.family: Theme.fontFamily
                                font.weight: Font.Bold
                                color: Theme.accent
                            }

                            // этап 0: описание правил
                            Text {
                                visible: wizard.stage === 0
                                Layout.fillWidth: true
                                text: qsTr("B ставим на первое слово сущности, I — на её продолжение, O — на служебные слова («для», «купить», «недорого»). Метка появляется над блоком. Backspace — снять метку и шагнуть назад, стрелки — навигация, Enter на последнем блоке — подтвердить этап.")
                                font.pixelSize: Theme.fontSmall
                                font.family: Theme.fontFamily
                                color: Theme.text
                                wrapMode: Text.WordWrap
                            }

                            // этап 1: список типов
                            Repeater {
                                model: wizard.stage === 1 ? wizard.categories.length : 0

                                Text {
                                    required property int index
                                    Layout.fillWidth: true
                                    text: (wizard.cats[wizard.entityIdx[wizard.cursor]] === wizard.categories[index] ? "▶ " : "   ")
                                          + (index + 1) + " · " + wizard.categories[index] + " — " + wizard.catShort[index]
                                    font.pixelSize: Theme.fontSmall
                                    font.family: Theme.fontFamily
                                    font.weight: wizard.cats[wizard.entityIdx[wizard.cursor]] === wizard.categories[index] ? Font.Bold : Font.Normal
                                    color: Theme.text
                                }
                            }

                            Text {
                                visible: wizard.stage === 1
                                Layout.fillWidth: true
                                Layout.topMargin: 4
                                text: {
                                    const cur = wizard.cats[wizard.entityIdx[wizard.cursor]] || "BRAND"
                                    const i = wizard.categories.indexOf(cur)
                                    return qsTr("Выбран %1: %2").arg(cur).arg(wizard.catFull[i >= 0 ? i : 0])
                                }
                                font.pixelSize: Theme.fontSmall
                                font.family: Theme.fontFamily
                                color: Theme.textSecondary
                                wrapMode: Text.WordWrap
                            }

                            // этап 2: список подтипов
                            Repeater {
                                model: wizard.stage === 2 ? wizard.subtypes.length : 0

                                Text {
                                    required property int index
                                    Layout.fillWidth: true
                                    text: (index === wizard.subChoice ? "▶ " : "   ")
                                          + (index + 1) + " · " + wizard.subtypes[index].ru
                                          + " (" + wizard.subtypes[index].code + ") — " + wizard.subtypes[index].desc
                                    font.pixelSize: Theme.fontSmall
                                    font.family: Theme.fontFamily
                                    font.weight: index === wizard.subChoice ? Font.Bold : Font.Normal
                                    color: Theme.text
                                }
                            }
                        }
                    }

                    // ручной ввод подтипа «другое»
                    TextField {
                        id: manualField
                        Layout.fillWidth: true
                        visible: false
                        placeholderText: qsTr("Введите свой подтип и нажмите Enter…")
                        font.pixelSize: Theme.fontBody
                        font.family: Theme.fontFamily
                        color: Theme.text
                        onAccepted: wizard.manualDone()

                        background: Rectangle {
                            radius: Theme.radiusSmall
                            color: Theme.surface
                            border.width: 2
                            border.color: Theme.tagAttr
                        }
                    }

                    // подсказка внизу
                    Text {
                        Layout.alignment: Qt.AlignHCenter
                        text: wizard.stage === 0
                              ? qsTr("B / I / O — тег · ←/→ — блоки · Backspace — снять и назад · Enter — подтвердить")
                              : wizard.stage === 1
                              ? qsTr("1–5 — тип · ↑/↓ — выбор · ←/→ — блоки · Enter — подтвердить · Backspace — назад")
                              : qsTr("1–9 или ↑/↓ — подтип · Enter — применить и дальше · Backspace — назад")
                        font.pixelSize: 10
                        font.family: Theme.fontFamily
                        color: Theme.textTertiary
                    }
                }
            }
        }

        // навигация по запросам
        RowLayout {
            Layout.fillWidth: true

            GhostButton {
                text: qsTr("← Предыдущий запрос")
                onClicked: wizard.prevQuery()
            }

            Item { Layout.fillWidth: true }

            GhostButton {
                text: qsTr("Следующий запрос →")
                onClicked: wizard.nextQuery()
            }
        }
    }

    // ---- диалоги
    Dialog {
        id: confirmDialog
        anchors.centerIn: parent
        modal: true
        width: 380
        padding: 22

        property string text
        property var acceptAction: function() {}

        background: Rectangle {
            radius: Theme.radiusLarge
            color: Theme.surface
            border.width: 1
            border.color: Theme.border
        }

        Overlay.modal: Rectangle { color: Qt.rgba(0, 0, 0, 0.35) }

        contentItem: ColumnLayout {
            spacing: 16

            Text {
                Layout.fillWidth: true
                text: confirmDialog.text
                font.pixelSize: Theme.fontMedium
                font.family: Theme.fontFamily
                font.weight: Font.DemiBold
                color: Theme.text
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 10

                AccentButton {
                    Layout.fillWidth: true
                    text: qsTr("Да, подтверждаю")
                    onClicked: {
                        confirmDialog.close()
                        confirmDialog.acceptAction()
                        wizard.forceActiveFocus()
                    }
                }

                GhostButton {
                    Layout.fillWidth: true
                    text: qsTr("Нет, ещё поправлю")
                    onClicked: {
                        confirmDialog.close()
                        wizard.forceActiveFocus()
                    }
                }
            }
        }
    }

    Dialog {
        id: infoDialog
        anchors.centerIn: parent
        modal: true
        width: 380
        padding: 22

        property string text

        background: Rectangle {
            radius: Theme.radiusLarge
            color: Theme.surface
            border.width: 1
            border.color: Theme.border
        }

        Overlay.modal: Rectangle { color: Qt.rgba(0, 0, 0, 0.35) }

        contentItem: ColumnLayout {
            spacing: 16

            Text {
                Layout.fillWidth: true
                text: infoDialog.text
                font.pixelSize: Theme.fontBody
                font.family: Theme.fontFamily
                color: Theme.text
                wrapMode: Text.WordWrap
            }

            AccentButton {
                Layout.fillWidth: true
                text: qsTr("Понятно")
                onClicked: {
                    infoDialog.close()
                    wizard.forceActiveFocus()
                }
            }
        }
    }
}
