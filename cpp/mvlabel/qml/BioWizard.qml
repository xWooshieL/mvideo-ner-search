// трёхэтапный мастер BIO-разметки: B/I/O -> тип сущности -> подтип атрибута
// тип ставится на сущность целиком (B I ... I — одна группа), фон (история) заблюрен
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
    property int cursor: 0           // активный блок (этап 0) или группа (этапы 1/2)
    property int subChoice: 0
    property bool wizardOpen: true   // false = окно разметки закрыто, виден фон с историей
    property var tokens: []
    property var bio: []             // "B"/"I"/"O"/""
    property var cats: []            // тип для каждого токена (внутри группы одинаковый)
    property var subs: []            // подтип для ATTR
    property var groups: []          // сущности: массив массивов индексов токенов
    property var attrGroups: []      // группы с типом ATTR (этап 2)

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

    // сущности из BIO: B открывает группу, соседний I продолжает её
    function computeGroups() {
        let g = []
        for (let i = 0; i < tokens.length; ++i) {
            const t = bio[i]
            if (t === "B") {
                g.push([i])
            } else if (t === "I") {
                if (g.length > 0 && g[g.length - 1][g[g.length - 1].length - 1] === i - 1)
                    g[g.length - 1].push(i)
                else
                    g.push([i])   // I без B — считаем отдельной сущностью
            }
        }
        groups = g
    }

    function computeAttrGroups() {
        let a = []
        for (let gi = 0; gi < groups.length; ++gi)
            if (cats[groups[gi][0]] === "ATTR")
                a.push(groups[gi])
        attrGroups = a
    }

    function groupText(g) {
        return g.map(i => tokens[i]).join(" ")
    }

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
        groups = []; attrGroups = []
        manualField.text = ""
        manualField.visible = false
        // ВАЖНО: без этого клавиатурный фокус мог "залипнуть" на скрытом manualField
        // (например, если ушли на новый запрос кликом мыши, не нажав Enter после
        // ручного ввода подтипа) — тогда B/I/O/цифры перестают что-либо делать.
        wizard.forceActiveFocus()
    }

    // восстановление последнего этапа сохранённого запроса (маленькая кнопка «назад»)
    function restoreStage() {
        const hasBio = bio.some(t => t !== "")
        if (!hasBio)
            return
        computeGroups()
        const hasCats = groups.length > 0 && groups.some(g => cats[g[0]] !== "")
        if (!hasCats) {
            stage = 0
            cursor = tokens.length - 1
            return
        }
        computeAttrGroups()
        if (attrGroups.length > 0) {
            stage = 2
            cursor = attrGroups.length - 1
            subChoice = 0
        } else {
            stage = 1
            cursor = groups.length - 1
        }
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
        bio = b
        computeGroups()
        const catPat = ["CATEGORY", "ATTR", "BRAND", "MODEL"]
        for (let gi = 0; gi < groups.length; ++gi)
            for (const i of groups[gi])
                c[i] = catPat[gi % catPat.length]
        cats = c
        if (demo === "stage2") {
            stage = 1
            cursor = 0
        } else if (demo === "stage3") {
            computeAttrGroups()
            if (attrGroups.length === 0 && groups.length > 0) {
                for (const i of groups[0])
                    c[i] = "ATTR"
                cats = c
                computeAttrGroups()
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
        computeGroups()
        if (groups.length === 0) { save(); return }
        // дефолтный тип для групп без типа
        let c = cats.slice()
        for (const g of groups)
            for (const i of g)
                if (c[i] === "") c[i] = "BRAND"
        cats = c
        stage = 1
        cursor = 0
    }

    // ---- этап 1: тип на всю сущность
    function setType(catIndex) {
        let c = cats.slice()
        for (const i of groups[cursor])
            c[i] = categories[catIndex]
        cats = c
        if (cursor < groups.length - 1)
            cursor++
    }

    function cycleType(d) {
        let c = cats.slice()
        const g = groups[cursor]
        let ci = categories.indexOf(c[g[0]] || "BRAND")
        ci = (ci + d + categories.length) % categories.length
        for (const i of g)
            c[i] = categories[ci]
        cats = c
    }

    function typeEnter() {
        if (cursor < groups.length - 1) {
            cursor++
            return
        }
        computeAttrGroups()
        if (attrGroups.length === 0) { save(); return }
        stage = 2
        cursor = 0
        subChoice = 0
    }

    // ---- этап 2: подтип на всю ATTR-группу
    function subEnter() {
        const st = subtypes[subChoice]
        if (st.code === "other") {
            manualField.visible = true
            manualField.forceActiveFocus()
            return
        }
        let s = subs.slice()
        for (const i of attrGroups[cursor])
            s[i] = st.code
        subs = s
        subNext()
    }

    function manualDone() {
        const txt = manualField.text.trim()
        if (txt.length === 0)
            return
        let s = subs.slice()
        for (const i of attrGroups[cursor])
            s[i] = txt
        subs = s
        manualField.text = ""
        manualField.visible = false
        wizard.forceActiveFocus()
        subNext()
    }

    function subNext() {
        if (cursor < attrGroups.length - 1) {
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

    // маленькая кнопка: назад на последнее состояние, а не на BIO
    function prevQueryState() {
        if (pos > 0) {
            pos--
            loadQuery()
            restoreStage()
        }
    }

    function nextQuery() {
        if (pos < LabelStore.queries.length - 1) { pos++; loadQuery() }
    }

    // ---- клавиатура (латиница и русская раскладка: b/и, i/ш, o/щ)
    focus: true
    Keys.onPressed: (event) => {
        if (!wizardOpen) {
            event.accepted = false
            return
        }
        if (manualField.visible) {
            if (event.key === Qt.Key_Escape) {
                manualField.visible = false
                event.accepted = true
            }
            return
        }
        const txt = (event.text || "").toLowerCase()
        event.accepted = true
        if (stage === 0) {
            if (event.key === Qt.Key_B || txt === "и") setBio("B")
            else if (event.key === Qt.Key_I || txt === "ш") setBio("I")
            else if (event.key === Qt.Key_O || txt === "щ") setBio("O")
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
            else if (event.key === Qt.Key_Right) cursor = Math.min(groups.length - 1, cursor + 1)
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
            else if (event.key === Qt.Key_Left) { if (cursor > 0) { cursor--; subChoice = 0 } }
            else if (event.key === Qt.Key_Right) { if (cursor < attrGroups.length - 1) { cursor++; subChoice = 0 } }
            else if (event.key === Qt.Key_Backspace) { stage = 1; cursor = 0 }
            else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) subEnter()
            else event.accepted = false
        }
    }

    function catColor(cat) {
        const i = categories.indexOf(cat)
        return i >= 0 ? catColors[i] : Theme.tagO
    }

    // токен входит в активную группу?
    function tokenActive(index) {
        if (stage === 0) return index === cursor
        const g = stage === 1 ? groups[cursor] : attrGroups[cursor]
        return g !== undefined && g.indexOf(index) >= 0
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

            // фоновая история — блюрим, пока открыт мастер
            AppCard {
                id: backdrop
                anchors.fill: parent
                color: Theme.surfaceAlt

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10

                        Text {
                            text: qsTr("История разметок (клик — открыть запрос)")
                            font.pixelSize: 11
                            font.family: Theme.fontFamily
                            font.weight: Font.Bold
                            color: Theme.textSecondary
                        }

                        Item { Layout.fillWidth: true }

                        // переключатель аккаунтов — доступен, когда мастер закрыт
                        Row {
                            visible: !wizard.wizardOpen
                            spacing: 6

                            Repeater {
                                model: [
                                    { key: "nikita", label: qsTr("Никита") },
                                    { key: "nekit", label: qsTr("Некит") },
                                    { key: "liza", label: qsTr("Лиза") }
                                ]

                                Rectangle {
                                    required property var modelData
                                    readonly property bool current: LabelStore.annotatorId === modelData.key
                                    width: accText.implicitWidth + 22
                                    height: 28
                                    radius: 7
                                    color: current ? Theme.accent : Theme.surface
                                    border.width: 1
                                    border.color: current ? Theme.accent : Theme.borderStrong

                                    Text {
                                        id: accText
                                        anchors.centerIn: parent
                                        text: parent.modelData.label
                                        font.pixelSize: Theme.fontSmall
                                        font.family: Theme.fontFamily
                                        font.weight: Font.Bold
                                        color: parent.current ? "#ffffff" : Theme.text
                                    }

                                    HoverHandler { cursorShape: Qt.PointingHandCursor }
                                    TapHandler {
                                        onTapped: {
                                            LabelStore.setAnnotator(parent.modelData.key)
                                            wizard.pos = LabelStore.firstUnlabeledBio()
                                            wizard.loadQuery()
                                        }
                                    }
                                }
                            }
                        }

                        AccentButton {
                            visible: !wizard.wizardOpen
                            text: qsTr("Продолжить разметку")
                            implicitHeight: 30
                            onClicked: {
                                wizard.wizardOpen = true
                                wizard.forceActiveFocus()
                            }
                        }
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
                                    wizard.wizardOpen = true
                                    wizard.forceActiveFocus()
                                }
                            }
                        }
                    }
                }

                layer.enabled: wizard.wizardOpen
                layer.effect: MultiEffect {
                    blurEnabled: true
                    blur: 0.85
                    blurMax: 24
                }
            }

            // передний план — мастер
            Rectangle {
                id: front
                visible: wizard.wizardOpen
                anchors.centerIn: parent
                width: Math.min(parent.width - 40, 920)
                height: Math.min(parent.height - 24, frontCol.implicitHeight + 44)
                radius: Theme.radiusLarge
                color: Qt.alpha(Theme.surface, 0.96)
                border.width: 1
                border.color: Theme.border

                // крестик — закрыть окно разметки (можно вернуться позже)
                Rectangle {
                    id: closeBtn
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.margins: 10
                    width: 26
                    height: 26
                    radius: 13
                    color: closeHover.hovered ? Theme.accentSoft : "transparent"
                    z: 5

                    Text {
                        anchors.centerIn: parent
                        text: "\u2715"
                        font.pixelSize: 12
                        font.weight: Font.Bold
                        color: closeHover.hovered ? Theme.accent : Theme.textSecondary
                    }

                    HoverHandler { id: closeHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: wizard.wizardOpen = false }
                }

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

                                readonly property bool isActive: wizard.tokenActive(index)
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
                                        return qsTr("Клавиши: B/и — начало сущности · I/ш — продолжение · O/щ — не сущность")
                                    if (wizard.stage === 1)
                                        return qsTr("Сущность «%1» — выберите тип (один на всю сущность):").arg(wizard.groups[wizard.cursor] !== undefined ? wizard.groupText(wizard.groups[wizard.cursor]) : "")
                                    return qsTr("Атрибут «%1» — выберите подтип:").arg(wizard.attrGroups[wizard.cursor] !== undefined ? wizard.groupText(wizard.attrGroups[wizard.cursor]) : "")
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
                                text: qsTr("B ставим на первое слово сущности, I — на её продолжение, O — на служебные слова («для», «купить», «недорого»). Работают обе раскладки: B/I/O и и/ш/щ. Backspace — снять метку и шагнуть назад, стрелки — навигация, Enter на последнем блоке — дальше.")
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
                                    readonly property string curCat: wizard.groups[wizard.cursor] !== undefined ? (wizard.cats[wizard.groups[wizard.cursor][0]] || "") : ""
                                    Layout.fillWidth: true
                                    text: (curCat === wizard.categories[index] ? "▶ " : "   ")
                                          + (index + 1) + " · " + wizard.categories[index] + " — " + wizard.catShort[index]
                                    font.pixelSize: Theme.fontSmall
                                    font.family: Theme.fontFamily
                                    font.weight: curCat === wizard.categories[index] ? Font.Bold : Font.Normal
                                    color: Theme.text
                                }
                            }

                            Text {
                                visible: wizard.stage === 1
                                Layout.fillWidth: true
                                Layout.topMargin: 4
                                text: {
                                    if (wizard.groups[wizard.cursor] === undefined) return ""
                                    const cur = wizard.cats[wizard.groups[wizard.cursor][0]] || "BRAND"
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
                              ? qsTr("B/и · I/ш · O/щ — тег · ←/→ — блоки · Backspace — снять и назад · Enter — дальше")
                              : wizard.stage === 1
                              ? qsTr("1–5 — тип сущности · ↑/↓ — выбор · ←/→ — сущности · Enter — дальше · Backspace — назад")
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

            Column {
                spacing: 4

                // маленькая кнопка: назад на последнее состояние (не на BIO)
                GhostButton {
                    text: "↩"
                    implicitHeight: 24
                    implicitWidth: 40
                    onClicked: wizard.prevQueryState()

                    ToolTip.visible: hovered
                    ToolTip.delay: 400
                    ToolTip.text: qsTr("Назад на последнее состояние разметки")
                }

                GhostButton {
                    text: qsTr("← Предыдущий запрос")
                    onClicked: wizard.prevQuery()
                }
            }

            Item { Layout.fillWidth: true }

            GhostButton {
                text: qsTr("Следующий запрос →")
                onClicked: wizard.nextQuery()
            }
        }
    }

    // ---- инфо-диалог (незаполненные блоки)
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
