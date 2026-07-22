cmake_minimum_required(VERSION 3.21)

project(MvSearchApp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Widgets Concurrent)

qt_standard_project_setup(REQUIRES 6.5)

qt_add_executable(mvsearch
    src/main.cpp
    src/searchengine.h src/searchengine.cpp
    assets/app.rc
)

set_source_files_properties(qml/Theme.qml PROPERTIES QT_QML_SINGLETON_TYPE TRUE)

qt_add_qml_module(mvsearch
    URI MvSearch
    VERSION 1.0
    QML_FILES
        qml/Main.qml
        qml/Theme.qml
        qml/SplashScreen.qml
        qml/MainWindow.qml
        qml/pages/SearchPage.qml
        qml/pages/StatsPage.qml
        qml/pages/SettingsPage.qml
        qml/components/AccentButton.qml
        qml/components/GhostButton.qml
        qml/components/AppCard.qml
        qml/components/AppTextField.qml
        qml/components/SideBarItem.qml
        qml/components/StatTile.qml
        qml/components/PageHeader.qml
        qml/components/TitleBar.qml
        qml/components/Toast.qml
        qml/components/FactChip.qml
    RESOURCES
        assets/logo_mark.png
        assets/logo_mark_white.png
        assets/icon.png
)

set_target_properties(mvsearch PROPERTIES
    WIN32_EXECUTABLE TRUE
    OUTPUT_NAME "MvSearch"
)

target_include_directories(mvsearch PRIVATE src)

target_link_libraries(mvsearch PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Widgets Qt6::Concurrent
)
