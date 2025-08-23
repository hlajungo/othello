#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
//#include <doctest.h>

#include <doctest/doctest.h>
#include <hello.h>

#include <iostream>
#include <sstream>
#include <streambuf>


TEST_CASE("Hello") {
    // 儲存原始 cout buffer
    std::streambuf* old_buf = std::cout.rdbuf();

    // 準備攔截的輸出流
    std::ostringstream captured_output;
    std::cout.rdbuf(captured_output.rdbuf()); // 將 cout 輸出轉向 stringstream

    // 呼叫被測函數
    hello_from_template();

    // 還原 cout，避免後續攔截
    std::cout.rdbuf(old_buf);

    // 比對輸出
    CHECK(captured_output.str() == "Hello from template!\n");
}


