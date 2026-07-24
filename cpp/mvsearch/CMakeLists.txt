cmake_minimum_required(VERSION 3.21)

project(MvSearchApp VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 Widgets Concurrent)

qt_standard_project_setup(REQUIRES 6.5)

set(APP_SOURCES
    src/main.cpp
    src/searchengine.h src/searchengine.cpp
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

qt_add_executable(mvsearch ${APP_SOURCES})

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
    MACOSX_BUNDLE TRUE
    MACOSX_BUNDLE_BUNDLE_NAME "MVideo SmartSearch"
    MACOSX_BUNDLE_GUI_IDENTIFIER "ru.mvideo.smartsearch"
    MACOSX_BUNDLE_ICON_FILE "icon.icns"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${PROJECT_VERSION}"
    OUTPUT_NAME "MvSearch"
)

target_include_directories(mvsearch PRIVATE src)

target_link_libraries(mvsearch PRIVATE
    Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::Widgets Qt6::Concurrent
)

# данные движка (словари, каталог, марковская модель) — рядом с exe (Win) или в Resources бандла (macOS).
# MV_REPO_ROOT можно переопределить (-DMV_REPO_ROOT=...), если проект собирается из копии
# без остального репозитория (например, зеркало на ASCII-путь на Windows).
if(NOT MV_REPO_ROOT)
    set(MV_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()
set(MV_DATA_CANDIDATES
    "${MV_REPO_ROOT}/artifacts/dicts/brands.txt"
    "${MV_REPO_ROOT}/artifacts/brands.txt"
    "${MV_REPO_ROOT}/artifacts/dicts/categories.txt"
    "${MV_REPO_ROOT}/artifacts/categories.txt"
    "${MV_REPO_ROOT}/artifacts/dicts/model_phrases.txt"
    "${MV_REPO_ROOT}/artifacts/model_phrases.txt"
    "${MV_REPO_ROOT}/artifacts/dicts/spell_aliases.txt"
    "${MV_REPO_ROOT}/models/markov_typer.json"
    "${MV_REPO_ROOT}/apps/mvp/catalog.json"
)
set(MV_DATA_FILES "")
set(MV_SEEN_NAMES "")
foreach(f ${MV_DATA_CANDIDATES})
    if(EXISTS "${f}")
        get_filename_component(_bn "${f}" NAME)
        list(FIND MV_SEEN_NAMES "${_bn}" _idx)
        if(_idx EQUAL -1)
            list(APPEND MV_SEEN_NAMES "${_bn}")
            list(APPEND MV_DATA_FILES "${f}")
        endif()
    endif()
endforeach()

if(MV_DATA_FILES)
    if(APPLE)
        set(MV_DATA_DEST "$<TARGET_FILE_DIR:mvsearch>/../Resources/data")
    else()
        set(MV_DATA_DEST "$<TARGET_FILE_DIR:mvsearch>/data")
    endif()
    add_custom_command(TARGET mvsearch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${MV_DATA_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MV_DATA_FILES} "${MV_DATA_DEST}"
        COMMENT "Копирую данные движка в ${MV_DATA_DEST}"
    )
else()
    message(STATUS "mvsearch: данные (artifacts/models/apps) не найдены в ${MV_REPO_ROOT} — копируй data/ вручную рядом с exe, либо задай -DMV_REPO_ROOT=<путь к репозиторию>")
endif()
