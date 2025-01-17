TEMPLATE = app

TARGET = TestCpp

CONFIG += c++20

SOURCES += \
        main.cpp

CONFIG += link_pkgconfig
PKGCONFIG += opencv4
