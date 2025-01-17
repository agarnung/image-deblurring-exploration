TEMPLATE = app

TARGET = TestCpp

CONFIG += c++20

SOURCES += \
        deconv_hyper_lap.cpp \
        main.cpp

HEADERS += \
        deconv_hyper_lap.h \

CONFIG += link_pkgconfig
PKGCONFIG += opencv4 fftw3

LIBS += -lfftw3 -lfftw3_threads -lfftw3f

HEADERS += \
    deconv_hyper_lap.h

QMAKE_CXXFLAGS += -Wall -O2
