CXX = g++
CXXFLAGS = -Wall -std=c++14 -static -Ofast -flto -march=native -mtune=native
LDLIBS = -lboost_program_options
HEADERS = $(shell find . -name "*.h" -o -name "*.hpp")
OBJDIR = obj
OBJS = $(OBJDIR)/main.o $(OBJDIR)/position.o $(OBJDIR)/tables.o $(OBJDIR)/types.o
TARGET = fengine
PREFIX = /usr/local

$(TARGET): $(OBJS)
	$(CXX) $^ $(CXXFLAGS) $(LDLIBS) -o $@

$(OBJDIR)/main.o: main.cpp $(HEADERS)
	mkdir -p $(OBJDIR)
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJDIR)/position.o: surge/src/position.cpp surge/src/*.h
	mkdir -p $(OBJDIR)
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJDIR)/tables.o: surge/src/tables.cpp surge/src/*.h
	mkdir -p $(OBJDIR)
	$(CXX) -c $< $(CXXFLAGS) -o $@

$(OBJDIR)/types.o: surge/src/types.cpp surge/src/*.h
	mkdir -p $(OBJDIR)
	$(CXX) -c $< $(CXXFLAGS) -o $@

.PHONY: clean install

clean:
	rm -rf $(TARGET) $(OBJDIR)

install:
	cp $(TARGET) $(PREFIX)/bin/
