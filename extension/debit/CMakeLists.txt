cmake_minimum_required(VERSION 3.16.0)

project(DebitExtension)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dbgen/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/combit/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/croaring)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ewah)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Concise)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")

add_compile_options(-mavx512f -mavx512bw -mavx512vbmi2 -march=native)

set(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/debit_extension.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit_adapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit/src/combit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit/src/and.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit/src/or.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit/src/xor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/combit/src/not.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/croaring/roaring.c
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/utils/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/fastbit/bitvector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/fastbit/array_t.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/fastbit/fileManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/fastbit/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wah_backend/src/fastbit/resource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dbgen/bm_dbgen.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dbgen/bmtpch_constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/bitmap_table_scan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q4.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q6.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q10.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q12.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q14.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q15.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q17.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/execution/tpch/query/Q19.cpp
)

build_static_extension(debit ${FILES})

build_loadable_extension(debit "" ${FILES})

#target_link_libraries(debit_extension pthread atomic ${Boost_LIBRARIES})

install(
  TARGETS debit_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)