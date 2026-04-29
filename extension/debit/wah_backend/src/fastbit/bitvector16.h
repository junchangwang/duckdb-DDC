#ifndef BITVECTOR16_H
#define BITVECTOR16_H

#include <stdio.h>
#include <iostream>
#include <boost/array.hpp>
#include "utils/util.h"
#include "bitvector_base.h"
#include <bitset>


class ibis::bitvector16 : public bitvector_base {
public :
    typedef uint16_t word_t;      /// the basic unit of data storage is 16-bit.
    
    // default construct 
    bitvector16() : nbits(0), nset(0), active(), m_vec() {};

    bitvector16(const bitvector16& bv);

    ~bitvector16() {
        clear();
    };

    void clear() {
        nbits = 0;
        nset = 0;
        active.reset();
        m_vec.clear();
    }

    void setBit(const uint64_t ind, int val, Table_config* config);

    int getBit(const uint64_t ind, Table_config* config) const;

    int getBitWithIndex(const uint64_t ind) const;  /// use in getBit function

    void append_active();                          /// append an active-word
    
    void append_fill(int val, word_t cnt);         /// 
    
    void appendWord(word_t w);                    /// use in setBit function

    void append_bits(int val, uint64_t n);

    void adjustSize(uint64_t nv, uint64_t nt);    /// Adjust the size of the bit sequence.

    /// the total number of bits in the bit sequence.
    uint64_t size() const {
        return ((nbits ? nbits : (nbits = do_cnt())) + active.nbits);
    }

    uint64_t Size() const {return static_cast<uint64_t>(size());}

    uint64_t getSerialSize() const throw() {
        return (m_vec.size() + 1 + (active.nbits > 0)) * sizeof(word_t);
    };

    void appendActive() {
    	if (active.nbits > 0)
            append_active();
    }
    
    bitvector_base& copy(const bitvector_base& bv){
        const bitvector16 &tmp = dynamic_cast<const bitvector16 &>(bv);
        clear();
        nbits = tmp.nbits;
        nset = tmp.nset;
        active = tmp.active;
        m_vec = tmp.m_vec;
        return *this;
    }

    /// Return the number of bits that are one.
    inline uint64_t cnt() const {
        if (nset == 0 && !m_vec.empty())
            nbits = do_cnt();
        return (nset + cnt_ones(active.val));
    }

    uint64_t count_ones() const {
        return static_cast<uint64_t>(cnt());
    }

    bitvector16& swap(bitvector16& bv) {
        uint64_t tmp;
        tmp = bv.nbits;
        bv.nbits = nbits;
        nbits = tmp;
        tmp = bv.nset;
        bv.nset = nset;
        nset = tmp;
        active_word atmp = bv.active;
        bv.active = active;
        active = atmp;
        m_vec.swap(bv.m_vec);
        return *this;
    }
    
    /// Count the number of bits and number of ones in m_vec. 
    uint64_t do_cnt() const throw();

    uint64_t cnt_ones(word_t val) const;

    // I/O function
    void write(const char* fn) const;

    void read(const char* fn);

    void compress();

    void decompress();

    void decompress(std::vector<word_t>& tmp) const;

    bool all0s() const {
        if (m_vec.empty()) {
            return true;
        }
        else if (m_vec.size() == 1) {
            return (m_vec[0] == 0 || (m_vec[0] >= HEADER0 && m_vec[0] < HEADER1));
        }
        else {
            return false;
        }
    }

    bool all1s() const {
        if (m_vec.size() == 1) {
            return (m_vec[0] == ALLONES || (m_vec[0] > HEADER1));
        }
        else {
            return false;
        }
    }
    
    bitvector16 &operator+=(int b);

    /// Perform bitwise AND between this bitvector and rhs.
    void operator&=(const bitvector16& rhs);

    void operator&=(const bitvector_base& rhs) {
        *this &= dynamic_cast<const bitvector16&>(rhs);
    }

    /// Perform bitwise AND between this bitvector and rhs, return
    /// the result as a new bitvector.
    /// bitvector16* operator&(const bitvector16&) const;

    /// Perform bitwise OR.
    void operator|=(const bitvector16& rhs);

    void operator|=(const bitvector_base& rhs) {
        *this |= dynamic_cast<const bitvector16&>(rhs);
    }

    /// Perform bitwise OR and return the result as a new bitvector.
    /// bitvector16* operator|(const bitvector16&) const;

    /// Perform bitwise exclusive or (XOR).
    void operator^=(const bitvector16& rhs);

    void operator^=(const bitvector_base& rhs) {
        *this ^= dynamic_cast<const bitvector16&>(rhs);
    }

    /// Perform bitwise XOR and return the result as a new bitvector.
    /// bitvector16* operator^(const bitvector16&) const;

    std::vector<uint16_t> decode(std::vector<uint16_t>& append_rowids, Table_config* config);
    
    std::vector<word_t> m_vec;             /// Store whole words.

    void buildIndex();

    // static members, constants to be used internally
    static const uint64_t MAXBITS;
    static const uint64_t SECONDBIT;
    static const word_t FILLBIT;
    static const word_t HEADER0;
    static const word_t HEADER1;
    static const word_t ALLONES;
    static const word_t MAXCNT;
    static const word_t UNDBIT;
    static const word_t FULLBIT0;
    static const word_t FULLBIT1;

    /// The struct active_word 
    struct active_word {
        word_t val;      // the value
        unsigned nbits;    // total number of bits

        // construct
        active_word() : val(0), nbits(0) { };

        void reset() {
            val = 0;
            nbits = 0;
        };

        int is_full() const {
            return (nbits >= MAXBITS);
        };

        /// Append a single bit.  The argument must be either 0 or 1.
        void append(int b) {
            val <<= 1;
            nbits++;
            val += b;
        };
    }; // struct active_word

    /// An internal struct used during logical operations to track the usage of fill words.
    struct run16 {
        int isFill;       /// whether this run is fill
        int fillBit;      /// 0 or 1-fill
        int isUndefined;  /// 
        word_t nWords;  /// the number of fill-blocks
        uint64_t nbits;
        std::vector<word_t>::const_iterator it;   /// const_iterator

        int wordPos;     /// record the current number of 15-blocks 
        int vecPos;      /// record the position in m-vec
        uint64_t bitPos;
        
        /// construct
        run16() : isFill(0), fillBit(0), nWords(0), it(0), wordPos(0), vecPos(-1), nbits(0), bitPos(0) { };
        ///  Decode the word pointed by  it.
        void decode() {
            fillBit = (*it > HEADER1);            /// 
            if (*it > ALLONES) {                  /// *it is a fill-word
                nWords = (*it & MAXCNT);
                nbits = nWords * MAXBITS;
                isFill = 1;
            }
            else {                               /// *it is a literal-word
                nWords = 1;
                nbits = MAXBITS;
                isFill = 0;
            }
            wordPos += nWords;
            vecPos += 1;
        };
        /// Reduce the run size by 1 word.
        void operator--() {
            if (nWords > 1) {
                --nWords;
            }
            else {
                ++it;
                nWords = 0;
            }
        };
        /// Reduce the run16 size by len words. 
        void operator-=(word_t len) {
            while (len > 0) {
                if (nWords == 0) decode();
                if (isFill != 0) {
                    if (nWords > len) {
                        nWords -= len;
                        len = 0;
                    }
                    else if (nWords == len) {
                        nWords = 0;
                        len = 0;
                        ++it;
                    }
                    else {
                        len -= nWords;
                        ++it;
                        nWords = 0;
                    }
                }
                else {
                    --len;
                    ++it;
                    nWords = 0;
                }
            }
        };
        /// Reduce the run16 size by len bits. 
        void operator -= (uint64_t len) {
            bitPos += len;
            while (len > 0) {
                if (nbits == 0) decode();
                if (nbits > len) {
                    nbits -= len;
                    len = 0;
                }
                else if (nbits == len) {
                    len = 0;
                    ++it;
                    nbits = 0;
                }
                else {
                    len -= nbits;
                    ++it;
                    nbits = 0;
                }
            }
        };
        
    };

    active_word active;       /// The active word.  
    mutable uint64_t nbits;   /// Number of bits in  m_vec.
    
    inline void copy_runs(run16& it, word_t& nw); // copy nw words
    void showindex(){            
        //for(auto it : m_vec) {
        //    cout << bitset<16>(it) << " " ;
        //}
        //std::cout << endl; 
        std::cout << "active info: " << active.nbits << " " << std::bitset<16> (active.val) << std::endl;
        do_cnt();
        std::cout << "read btv16: "<< nbits << " " << nset << std::endl << std::endl;      
    }
    
private:

    friend struct run16;          
    friend struct active_word;

    // member variables of bitvector class
    mutable uint64_t nset;    /// Number of bits that are 1 in  m_vec.

    // The following three functions all performs or operation, _c2 and _c1
    // generate compressed solutions, but _c0, _d1, and _d2 generate
    // uncompressed solutions.
    // or_c2 assumes there are compressed word in both operands
    // or_c1 *this may contain compressed word, but not rhs
    // or_c0 assumes both operands are not compressed
    // or_d1 *this contains no compressed word and is overwritten with the
    //       result
    // or_d2 both *this and rhs are compressed, but res is not compressed
    void or_c2(const bitvector16& rhs, bitvector16& res) const;

    void or_c1(const bitvector16& rhs, bitvector16& res) const;

    void or_c0(const bitvector16& rhs);

    void or_d1(const bitvector16& rhs);

    void or_d2(const bitvector16& rhs, bitvector16& res) const;

    void and_c2(const bitvector16& rhs, bitvector16& res) const;

    void and_c1(const bitvector16& rhs, bitvector16& res) const;

    void and_c0(const bitvector16& rhs);

    void and_d1(const bitvector16& rhs);

    void and_d2(const bitvector16& rhs, bitvector16& res) const;

    void xor_c2(const bitvector16& rhs, bitvector16& res) const;

    void xor_c1(const bitvector16& rhs, bitvector16& res) const;

    void xor_c0(const bitvector16& rhs);

    void xor_d1(const bitvector16& rhs);

    void xor_d2(const bitvector16& rhs, bitvector16& res) const;

    void copy_comp(std::vector<word_t>& tmp) const;

    inline void copy_fill(std::vector<word_t>::iterator& jt, run16& it);

    inline void copy_runsn(run16& it, word_t& nw); // copy nw words and negate

    inline void copy_runs(std::vector<word_t>::iterator& jt, run16& it,
        word_t& nw);

    inline void copy_runsn(std::vector<word_t>::iterator& jt, run16& it,
        word_t& nw);

    inline void copy_runs(run16& it, word_t& nw, const std::vector<std::pair<int, uint64_t>>& index, int& bi_idx_pos); // copy nw words
};

#endif



/// Copy a group of consecutive runs.  It appends nw words starting from
/// 'it' to the current bit vector, assuming active is empty.  Both it and
/// nw are modified in this function.  On returning from this function, it
/// points to the next unused word and nw stores the value of remaining
/// words to copy.
inline void ibis::bitvector16::copy_runs(run16& it, word_t& nw) {
    // deal with the first word -- attach it to the last word in m_vec
    if (it.isFill != 0) {
        if (it.nWords > 1) {
            append_fill(it.fillBit, it.nWords);
            nw -= it.nWords;
        }
        else if (it.nWords == 1) {
            active.val = (it.fillBit != 0 ? ALLONES : 0);
            append_active();
            --nw;
        }
    }
    else {
        active.val = *(it.it);
        append_active();
        --nw;
    }
    ++it.it;
    nset = 0;
    it.nWords = 0;
    nbits += MAXBITS * nw;

    while (nw > 0) { // copy the words
        it.decode();
        if (nw >= it.nWords) {
            m_vec.push_back(*(it.it));
            nw -= it.nWords;
            it.nWords = 0;
            ++it.it;
        }
        else {
            break;
        }
    }
    nbits -= MAXBITS * nw;
} // ibis::bitvector16::copy_runs

/// Copy the complements of a set of consecutive runs.  It assumes
/// active to be empty.
inline void ibis::bitvector16::copy_runsn(run16& it, word_t& nw) {
    // deal with the first word -- need to attach it to the last word in m_vec
    if (it.isFill != 0) {
        if (it.nWords > 1) {
            append_fill(!it.fillBit, it.nWords);
            nw -= it.nWords;
        }
        else if (it.nWords == 1) {
            active.val = (it.fillBit != 0 ? 0 : ALLONES);
            append_active();
            --nw;
        }
    }
    else {
        active.val = ALLONES ^ *(it.it);
        append_active();
        --nw;
    }
    ++it.it; // advance to the next word
    nset = 0;
    it.nWords = 0;
    nbits += MAXBITS * nw;

    while (nw > 0) { // copy the words
        it.decode();
        if (nw >= it.nWords) {
            m_vec.push_back((it.isFill ? FILLBIT : ALLONES) ^ *(it.it));
            nw -= it.nWords;
            it.nWords = 0;
            ++it.it;
        }
        else {
            break;
        }
    }
    nbits -= MAXBITS * nw;
} // ibis::bitvector16::copy_runsn

/// Copy the fill in "run it" as literal words.
/// This implementation relies on the fact that the iterator jt is actually
/// a bare pointer.
inline void ibis::bitvector16::copy_fill
(std::vector<ibis::bitvector16::word_t>::iterator& jt, run16& it) {
    if (it.fillBit == 0) {
        while (it.nWords > 3) {
            *jt = 0;
            jt[1] = 0;
            jt[2] = 0;
            jt[3] = 0;
            jt += 4;
            it.nWords -= 4;
        }
        if (it.nWords == 1) {
            *jt = 0;
            ++jt;
        }
        else if (it.nWords == 2) {
            *jt = 0;
            jt[1] = 0;
            jt += 2;
        }
        else if (it.nWords == 3) {
            *jt = 0;
            jt[1] = 0;
            jt[2] = 0;
            jt += 3;
        }
    }
    else {
        while (it.nWords > 3) {
            *jt = ALLONES;
            jt[1] = ALLONES;
            jt[2] = ALLONES;
            jt[3] = ALLONES;
            jt += 4;
            it.nWords -= 4;
        }
        if (it.nWords == 1) {
            *jt = ALLONES;
            ++jt;
        }
        else if (it.nWords == 2) {
            *jt = ALLONES;
            jt[1] = ALLONES;
            jt += 2;
        }
        else if (it.nWords == 3) {
            *jt = ALLONES;
            jt[1] = ALLONES;
            jt[2] = ALLONES;
            jt += 3;
        }
    }
    it.nWords = 0;
    ++it.it;
} // ibis::bitvector16::copy_fill

/// Copy the next nw words (nw * MAXBITS bits) starting with run it
/// to an array_t as uncompressed words.  If the run has more words than nw,
/// return the left over words to give it a chance for the longer run to be
/// counted first.
inline void ibis::bitvector16::copy_runs
(std::vector<ibis::bitvector16::word_t>::iterator& jt, run16& it, word_t& nw) {
    while (nw >= it.nWords && nw > 0) {
        if (it.isFill != 0) { // copy a fill
            const std::vector<word_t>::iterator iend = jt + it.nWords;
            if (it.fillBit == 0) {
                while (jt < iend) {
                    *jt = 0;
                    ++jt;
                }
            }
            else {
                while (jt < iend) {
                    *jt = ALLONES;
                    ++jt;
                }
            }
            nw -= it.nWords;
        }
        else { // copy a single word
            *jt = *(it.it);
            ++jt;
            --nw;
        }
        ++it.it; // advance to the next word
        if (nw > 0) {
            it.decode();
        }
        else {
            it.nWords = 0;
            return;
        }
    }
} // ibis::bitvector16::copy_runs

/// Copy the complements of the next nw words (nw * MAXBITS bits)
/// starting with "run it" as uncompressed words.
inline void ibis::bitvector16::copy_runsn
(std::vector<ibis::bitvector16::word_t>::iterator& jt, run16& it, word_t& nw) {
    while (nw >= it.nWords) {
        if (it.isFill != 0) { // a fill
            const std::vector<word_t>::iterator iend = jt + it.nWords;
            if (it.fillBit == 0) {
                while (jt < iend) {
                    *jt = ALLONES;
                    ++jt;
                }
            }
            else {
                while (jt < iend) {
                    *jt = 0;
                    ++jt;
                }
            }
            nw -= it.nWords;
        }
        else { // a literal word
            *jt = *(it.it) ^ ALLONES;
            ++jt;
            --nw;
        }
        ++it.it; // advance to the next word
        if (nw > 0) {
            it.decode();
        }
        else {
            it.nWords = 0;
            return;
        }
    }
} // ibis::bitvector16::copy_runsn

inline void ibis::bitvector16::copy_runs(run16& it, word_t& nw, const std::vector<std::pair<int, uint64_t>>& index,
    int& bi_idx_pos) {
    // deal with the first word -- attach it to the last word in m_vec
    if (it.isFill != 0) {
        if (it.nWords > 1) {
            append_fill(it.fillBit, it.nWords);
            nw -= it.nWords;
        }
        else if (it.nWords == 1) {
            active.val = (it.fillBit != 0 ? ALLONES : 0);
            append_active();
            --nw;
        }
    }
    else {
        active.val = *(it.it);
        append_active();
        --nw;
    }
    ++it.it;
    nset = 0;
    it.nWords = 0;
    nbits += MAXBITS * nw;

    int bi_idx_pos_old = bi_idx_pos;
    // if (bi_idx_pos < index.size() && nw > 100)
    if (bi_idx_pos < index.size() && nw > 100) {
        int bi_copy_end = it.wordPos + nw;
        while (bi_idx_pos < index.size() && bi_copy_end > index[bi_idx_pos].second) {
            bi_idx_pos += 1;
        }
        if (bi_idx_pos_old < bi_idx_pos) {
            bi_idx_pos -= 1;
            int copyVec = index[bi_idx_pos].first - it.vecPos;
            if (copyVec > 0 && index[bi_idx_pos].second <= bi_copy_end) {
                for (int i = 0; i < copyVec; ++i, ++it.it) {
                    *m_vec.end() = *(it.it);
                    ++m_vec.end();
                }
                nw -= index[bi_idx_pos].second - it.wordPos;
                it.vecPos = index[bi_idx_pos].first;
                it.wordPos = index[bi_idx_pos].second;
            }
        }
    }

    while (nw > 0) { // copy the words
        it.decode();
        if (nw >= it.nWords) {
            m_vec.push_back(*(it.it));
            nw -= it.nWords;
            it.nWords = 0;
            ++it.it;
        }
        else {
            break;
        }
    }
    nbits -= MAXBITS * nw;
} // ibis::bitvector16::copy_runs


