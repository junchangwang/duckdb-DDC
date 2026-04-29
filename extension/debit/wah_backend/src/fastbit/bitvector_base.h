#ifndef BITVECTOR_BASE_H
#define BITVECTOR_BASE_H
/*
bitvector_base.h is for the base bitvetcor class

!!!NOTE
bitvector8.h  -> 8-bit bitvector class
bitvector16.h -> 16-bit bitvector class
bitvector.h   -> 32-bit bitvector class
bitvector64.h -> 64-bit bitvector class
*/

#include <iostream>
#include <boost/array.hpp>
#include "fastbit/array_t.h"	// alternative to std::vector
#include "utils/util.h"

#define MAKE_BITVECTOR(ptr, x) \
    switch (x) \
    { \
    case 8: \
        ptr = new ibis::bitvector8(); \
        break; \
    case 16: \
        ptr = new ibis::bitvector16(); \
        break; \
    case 32: \
        ptr = new ibis::bitvector(); \
        break; \
    default: \
        assert(0); \
        break; \
    } \

class bitvector_base {
public:
	bitvector_base() {}
	
	virtual ~bitvector_base() {};

	virtual void setBit(const uint64_t ind, int val, Table_config* config) {return;}

	virtual int getBit(const uint64_t ind, Table_config* config) const {return -1;}

	virtual int getBitWithIndex(const uint64_t ind) const {return -1;}

	//virtual void append_active() ;

	//virtual void append_fill(int val, word_t cnt) = 0;

	//virtual void appendWord(word_t w) = 0;

	virtual void adjustSize(uint64_t nv, uint64_t nt) {return;}

	virtual uint64_t do_cnt() const throw() {return -1;}
	virtual uint64_t count_ones() const {return -1;}

    /// the total number of bits in the bit sequence.
    virtual uint64_t Size() const {return -1;}

	//virtual uint64_t cnt_ones(word_t val) const = 0;

	virtual void write(const char* fn) const {return;}

	virtual void read(const char* fn) {return;}

	virtual void compress() {return;}

	virtual void decompress() {return;}
	
	virtual void buildIndex() {return;}
	
	virtual uint64_t getSerialSize() const throw() {return -1;}
	
	virtual void appendActive() {return;}

	virtual bitvector_base& copy(const bitvector_base& rhs) = 0;
	virtual void operator^=(const bitvector_base& rhs) {return;}
	virtual void operator&=(const bitvector_base& rhs) {return;}
	virtual void operator|=(const bitvector_base& rhs) {return;}

	
	
//private:
       std::vector<std::pair<int, uint64_t>> index;/// index.first: like it
                                           /// index.second: the number of maxmium-block

};

#endif
