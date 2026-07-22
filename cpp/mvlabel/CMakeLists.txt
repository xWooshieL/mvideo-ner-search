cmake_minimum_required(VERSION 3.21)

project(MvLabelApp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Widgets)

qt_standard_project_setup(REQUIRES 6.5)

set(APP_SOURCES
    src/main.cpp
    src/labelstore.h src/labelstore.cpp
)
if(WIN32)
    list(APPEND APP_SOURCES assets/app.rc)
endif()
if(APPLE)
    set(MACOSX_ICON assets/icon.icns)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${MACOSX_ICON}")
        list(APPEND APP_SOURCES ${MACOSX_ICON})
        set_source_files_properties(${MACOSX_ICON} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources")
    endif()
endif()

qt_add_executable(mvlabel ${APP_SOURCES})

set_source_files_properties(qml/Theme.qml PROPERTIES QT_QML_SINGLETON_TYPE TRUE)

qt_add_qml_module(mvlabel
    URI MvLabel
    VERSION 1.0
    QML_FILES
        qml/Main.qml
        qml/Theme.qml
        qml/SplashScreen.qml
        qml/MainWindow.qml
        qml/BioWizard.qml
        qml/MatchPage.qml
        qml/SettingsPopup.qml
        qml/components/AccentButton.qml
        qml/components/GhostButton.qml
        qml/components/AppCard.qml
        qml/components/TitleBar.qml
        qml/components/NiceProgress.qml
    RESOURCES
        assets/logo_mark.png
        assets/logo_mark_white.png
        assets/icon.png
)

set_target_properties(mvlabel PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "MVideo Labeling"
    MACOSX_BUNDLE_GUI_IDENTIFIER "ru.mvideo.labeling"
    MACOSX_BUNDLE_ICON_FILE "icon.icns"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    OUTPUT_NAME "MvLabel"
)

target_include_directories(mvlabel PRIVATE src)

target_link_libraries(mvlabel PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Widgets
)

# данные разметки (запросы и пары для Никиты/Некита/Лизы) — рядом с exe (Win) или в Resources (macOS).
# MV_REPO_ROOT можно переопределить (-DMV_REPO_ROOT=...), см. комментарий в cpp/mvsearch/CMakeLists.txt.
if(NOT MV_REPO_ROOT)
    set(MV_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()
file(GLOB MV_LABEL_DATA_FILES "${MV_REPO_ROOT}/apps/labeling/data/*.json")

if(MV_LABEL_DATA_FILES)
    if(APPLE)
        set(MV_LABEL_DATA_DEST "$<TARGET_FILE_DIR:mvlabel>/../Resources/data")
    else()
        set(MV_LABEL_DATA_DEST "$<TARGET_FILE_DIR:mvlabel>/data")
    endif()
    add_custom_command(TARGET mvlabel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MV_LABEL_DATA_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MV_LABEL_DATA_FILES} "${MV_LABEL_DATA_DEST}"
        COMMENT "Копирую данные разметки в ${MV_LABEL_DATA_DEST}"
    )
else()
    message(STATUS "mvlabel: apps/labeling/data/*.json не найдены в ${MV_REPO_ROOT} — копируй data/ вручную рядом с exe, либо задай -DMV_REPO_ROOT=<путь к репозиторию>")
endif()
