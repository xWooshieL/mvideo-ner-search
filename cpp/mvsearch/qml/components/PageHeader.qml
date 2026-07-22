// заголовок страницы с подзаголовком
import QtQuick
import MvSearch

Column {
    property string title
    property string subtitle

    spacing: 4

    Text {
        text: title
        font.pixelSize: Theme.fontTitle
        font.family: Theme.fontFamily
        font.weight: Font.Bold
        color: Theme.text
    }

    Text {
        text: subtitle
        visible: subtitle.length > 0
        font.pixelSize: Theme.fontBody
        font.family: Theme.fontFamily
        color: Theme.textSecondary
    }
}
