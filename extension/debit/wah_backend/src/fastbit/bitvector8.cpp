#include "utils/util.h"
#include <iostream>
#include "bitvector8.h"
#include "bitvector_base.h"
#include <iomanip>	// setw
#include<stdio.h>
#include<bitset>
#include<stdlib.h>
#include<ctime>
#include<fstream>

int test_cnt = 0;

unsigned int INDEX_WORDS_2 = 12222;

// static members
const uint64_t ibis::bitvector8::MAXBITS = 7;
const uint64_t ibis::bitvector8::SECONDBIT = 6;
const ibis::bitvector8::word_t ibis::bitvector8::ALLONES =    /// 01111111
(((word_t)1 << ibis::bitvector8::MAXBITS) - 1);
const ibis::bitvector8::word_t ibis::bitvector8::MAXCNT =     /// 00111111  
(((word_t)1 << ibis::bitvector8::SECONDBIT) - 1);
const ibis::bitvector8::word_t ibis::bitvector8::FILLBIT =    /// 01000000
((word_t)1 << bitvector8::SECONDBIT);
const ibis::bitvector8::word_t ibis::bitvector8::HEADER0 =    /// 10000000
((word_t)2 << ibis::bitvector8::SECONDBIT);
const ibis::bitvector8::word_t ibis::bitvector8::HEADER1 =    /// 11000000
((word_t)3 << ibis::bitvector8::SECONDBIT);
const ibis::bitvector8::word_t ibis::bitvector8::FULLBIT0 = 191;   /// 10111111
const ibis::bitvector8::word_t ibis::bitvector8::FULLBIT1 = 255;   /// 11111111



/// Copy constructor. 
ibis::bitvector8::bitvector8(const bitvector8& bv)
    : nbits(bv.nbits), nset(bv.nset), active(bv.active), m_vec(bv.m_vec) {
}

/// Replace a single bit at position ind with val.
void ibis::bitvector8::setBit(const uint64_t ind, int val, Table_config* config) {
    
    //std::cout << "do 8-bit serbit" << std::endl;
    if (ind >= size()) {
        uint64_t diff = ind - size() + 1;   /// Number of bits exceeded
        //std::cout << "diff:" << diff << std::endl;
        //std::cout << "size(): " << size() << std::endl;
        //std::cout << "active: " << active.nbits << std::endl;
        //std::cout << "nbits: " << nbits << std::endl << std::endl;
        if (active.nbits) {
            if (ind + 1 >= nbits + MAXBITS) {             /// the bit to set isn't in the active-word
                diff -= (MAXBITS - active.nbits);         /// the left bits after filling the active
                active.val <<= (MAXBITS - active.nbits);  /// Exceeding bits are automatically filled with 0
                if (diff == 0)    /// just get the ind location 
                    active.val += (val != 0);
                append_active();
            }
            else { /// the bit to set is in the active-wordn
                active.nbits += diff;
                active.val <<= diff;
                active.val += (val != 0);
                diff = 0;
            }
        }

        if (diff) {
            uint64_t w = diff / MAXBITS;    /// the number of MAXBITS-blocks that the left bits can fill 
            diff -= w * MAXBITS;            /// left bits less than MAXBITS
            if (diff) {
                while (w >= MAXCNT) {
                    append_fill(0, MAXCNT);   /// append w 0-fill words
                    w -= MAXCNT;
                }
                if (w > 1)
                    append_fill(0, w);
                else if (w)
                    append_active();
                active.nbits = diff;
                active.val += (val != 0);
            }
            /// diff == 0 means the bit is the last bit in the wth block
            else if (val != 0) {
                while (w >= MAXCNT) {
                    append_fill(0, MAXCNT);
                    w -= MAXCNT;
                }
                if (w > 2)
                    append_fill(0, w - 1);
                else if (w == 2)
                    append_active();
                active.val = 1;
                append_active();
            }
            else {
                while (w >= MAXCNT) {
                    append_fill(0, MAXCNT);   /// append w 0-fill words
                    w -= MAXCNT;
                }
                if (w > 1)
                    append_fill(0, w);
                else
                    append_active();
            }
        }
        if (nset)
            nset += (val != 0);
        return;
    }
    else if (ind >= nbits) { // modify an active bit
        word_t u = active.val;
        if (val != 0) {
            active.val |= ((word_t)1 << (active.nbits - (ind - nbits) - 1));
        }
        else {
            active.val &= ~((word_t)1 << (active.nbits - (ind - nbits) - 1));
        }
        if (nset && (u != active.val))
            nset += (val ? 1 : -1);
        return;
    }

    // uncompressed
    if (m_vec.size() * MAXBITS == nbits) {
        const uint64_t i = ind / MAXBITS;
        const word_t u = m_vec[i];
        const word_t w = ((word_t)1 << (SECONDBIT - (ind % MAXBITS)));
        if (val != 0)
            m_vec[i] |= w;                        /// need to overload |=
        else
            m_vec[i] &= ~w;                       /// need to overload &=
        if (nset && (m_vec[i] != u))
            nset += (val ? 1 : -1);
    }
    else {
        // compressed bitvector - the bit to be modified is in m_vec
        std::vector<word_t>::iterator it = m_vec.begin();
        int compressed;
        int cnt = 0;
        uint64_t ind1 = 0;
        uint64_t ind0 = ind;
        int current = 0; /// current bit value 

        /// struct timeval before, after;   /// test time

        if (config->enable_fence_pointer && index.size() > 0) {   /// use it to locate it faster
            int indPos = -1;
            int vecPos = ind / MAXBITS;
            for (int i = 0; i < index.size(); ++i) {
                indPos = i - 1;
                if (index[i].second >= vecPos) {
                    break;
                }
            }
            if (indPos >= 0) {
                ind0 -= index[indPos].second * MAXBITS;
                it += index[indPos].first + 1;
            }
        }

        /// use while to search the correct location
        while ((ind0 > 0) && (it < m_vec.end())) {
            if (*it >= HEADER0) {                // a fill word
                cnt = ((*it) & MAXCNT) * MAXBITS;// the number of fill-block in this word
                if (cnt > ind0) {                // found the location
                    current = (*it >= HEADER1);  // get the fill-number in this fill-word 
                    compressed = 1;
                    ind1 = ind0;
                    ind0 = 0;
                }
                else {
                    ind0 -= cnt;
                    ind1 = ind0;
                    ++it;
                }
            }
            else {    // a literal word
                cnt = MAXBITS;
                if (MAXBITS > ind0) {    // found the location
                    current = ((word_t)1 & ((*it) >> (SECONDBIT - ind0)));
                    compressed = 0;
                    ind1 = ind0;
                    ind0 = 0;
                }
                else {
                    ind0 -= MAXBITS;
                    ind1 = ind0;
                    ++it;
                }
            }
        } // while (ind...

        if (ind1 == 0) {    // set current and compressed
            if (*it >= HEADER0) {  // in a fill
                cnt = ((*it) & MAXCNT) * MAXBITS;
                current = (*it >= HEADER1);
                compressed = 1;
            }
            else {       // the bit to set is the first bit in this literal
                cnt = MAXBITS;
                current = (*it >> SECONDBIT);
                compressed = 0;
            }
        }

        if (ind0 > 0) { // has not found the right location yet
            if (ind0 < active.nbits) {  // in the active word
                word_t w = ((word_t)1 << (active.nbits - ind0 - 1));
                if (val != 0) {
                    active.val |= w;
                }
                else {
                    active.val &= ~w;
                }
            }
            else {     // extends the current bit vector
                ind1 = ind0 - active.nbits - 1;               /// leave temporarily
                appendWord(HEADER0 | (ind1 / MAXBITS));       /// ***** when ind1/MAXBITS > 16383
                for (ind1 %= MAXBITS; ind1 > 0; --ind1)
                    operator+=(0);
                operator+=(val != 0);                         /// need to overload +=
                if (nset) nset += val ? 1 : -1;
                return;
            }
        }

        // locate the bit to be changed, lots of work hidden here
        if (current == val)
            return; // nothing to do

        // need to actually modify the bit
        if (compressed == 0) { // toggle a single bit of a literal word
            *it ^= ((word_t)1 << (SECONDBIT - ind1));      /// need to overload ^= 
        }
        else if (ind1 < MAXBITS) {
            // bit to be modified is in the first word, two pieces
            --(*it);
            if ((*it & MAXCNT) == 1)
                *it = (current) ? ALLONES : 0;
            word_t w = ((word_t)1 << (SECONDBIT - ind1));
            if (val == 0) w ^= ALLONES;
            it = m_vec.insert(it, w);   // append a word w in front of it
        }
        else if (cnt - ind1 <= MAXBITS) {
            // bit to be modified is in the last word, two pieces
            --(*it);
            if ((*it & MAXCNT) == 1)
                *it = (current) ? ALLONES : 0;
            word_t w = ((word_t)1 << (cnt - ind1 - 1));
            if (val == 0) w ^= ALLONES;
            ++it;
            it = m_vec.insert(it, w);
        }
        else { // the counter breaks into three pieces
            word_t u[2], w;
            uint64_t nn = ind1 / MAXBITS;
            w = (*it & MAXCNT) - nn - 1;
            u[1] = 1 << (SECONDBIT - ind1 + nn * MAXBITS);
            if (val == 0) {
                u[0] = (nn > 1) ? (HEADER1 | nn) : (ALLONES); /// use u[0] to get first fill   *********
                u[1] ^= ALLONES;                                   /// use u[1] to create the new literal 
                w = (w > 1) ? (HEADER1 | w) : (ALLONES);           /// the second fill
            }
            else {
                u[0] = (nn > 1) ? (HEADER0 | nn) : static_cast<word_t>(0);
                w = (w > 1) ? (HEADER0 | w) : static_cast<word_t>(0);
            }
            *it = w;
            m_vec.insert(it, u, u + 2);   /// insert u[0] and u[1] before it
        }
        if (nset)
            nset += val ? 1 : -1;
    }
}

/// (okay)
int ibis::bitvector8::getBit(const uint64_t ind, Table_config* config) const {
    //nbits = do_cnt();
    if (ind >= size()) {
        return 0;
    }
    else if (ind >= nbits) {  /// in the active
        return ((active.val >> (active.nbits - (ind - nbits) - 1)) & (word_t)1);
        
    }
    else if (m_vec.size() * MAXBITS == nbits) {  // uncompressed
        return ((m_vec[ind / MAXBITS] >> (SECONDBIT - (ind % MAXBITS))) & (word_t)1);
    }
    else { // need to decompress the compressed words
        if (config->enable_fence_pointer && index.size() > 0) {
            return getBitWithIndex(ind);     /// modify getBitWithIndex function
        }
        else {
            uint64_t jnd = ind;
            std::vector<word_t>::const_iterator it = m_vec.begin();
            while (it != m_vec.end()) {
                if (*it > HEADER0) { // a fill
                    const uint16_t cnt = ((*it) & MAXCNT) * MAXBITS;
                    if (cnt > jnd) {
                        return (*it >= HEADER1);
                    }
                    jnd -= cnt;
                }
                else if (jnd < MAXBITS) {
                    return ((*it >> (SECONDBIT - jnd)) & (word_t)1);
                }
                else {
                    jnd -= MAXBITS;
                }
                ++it;
            }
        }
    }
    return 0;
} // ibis::bitvector::getBit

/// use in getBit function (OKAY)
int ibis::bitvector8::getBitWithIndex(const uint64_t ind) const {
    uint64_t jnd = ind;
    std::vector<word_t>::const_iterator it = m_vec.begin();

    int indPos = -1;
    int vecPos = ind / MAXBITS;
    for (int i = 0; i < index.size(); ++i) {
        indPos = i - 1;
        if (index[i].second >= vecPos) {
            break;
        }
    }
    if (indPos >= 0) {
        jnd -= index[indPos].second * MAXBITS;
        it += index[indPos].first + 1;
    }

    while (it < m_vec.end()) {
        if (*it > HEADER0) {   // a fill
            const uint64_t cnt = ((*it) & MAXCNT) * MAXBITS;
            if (cnt > jnd) {
                return (*it >= HEADER1);
            }
            jnd -= cnt;
        }
        else if (jnd < MAXBITS) {
            return ((*it >> (SECONDBIT - jnd)) & (word_t)1);
        }
        else {
            jnd -= MAXBITS;
        }
        ++it;
    }
    return 0;
} // ibis::bitvector::getBit

/// append a single word(use in setBit function )
void ibis::bitvector8::appendWord(word_t w) {
    word_t nb1, nb2;
    word_t cps = (w >> MAXBITS);      /// type-bit
    //nset = 0;
    if (active.nbits) { // active contains some uncompressed bits
        word_t w1;
        nb1 = active.nbits;
        nb2 = MAXBITS - active.nbits;
        active.val <<= nb2;
        if (cps != 0) {                  // incoming bits are comporessed
            int b2 = (w >= HEADER1);     // fill-number bit
            if (b2 != 0) {               // 1-fill
                w1 = (1 << nb2) - 1;
                active.val |= w1;        // set the left bits to 1
            }
            append_active();
            nb2 = (w & MAXCNT) - 1;
            if (nb2 > 1) {        // append a counter
                append_fill(b2, nb2);
            }
            else if (nb2 == 1) {
                if (b2 != 0) active.val = ALLONES;
                append_active();
            }
            active.nbits = nb1;
            active.val = ((1 << nb1) - 1) * b2;
        }
        else { // incoming bits are not compressed
            w1 = (w >> nb1);
            active.val |= w1;
            append_active();
            w1 = (1 << nb1) - 1;
            active.val = (w & w1);
            active.nbits = nb1;
        }
    } // end of the case where there are active bits
    else if (cps != 0) { // no active bit and compressed
        int b2 = (w >= HEADER1);
        nb2 = (w & MAXCNT);
        if (nb2 > 1)
            append_fill(b2, nb2);
        else if (nb2 == 1) {
            if (b2)
                active.val = ALLONES;
            append_active();
        }
    }
    else { // no active bits and uncompressed
           // new word is a raw bit pattern, simply add the word
        active.val = w;
        append_active();
    }
} // ibis::bitvector::appendWord

/// push last full active into m_vec and append a new active (OKAY)
void ibis::bitvector8::append_active() {
    if (m_vec.empty()) {
        m_vec.push_back(active.val);
    }
    else if (active.val == 0) {           // incoming word is zero
        if (m_vec.back() == 0) {          // previous and incoming are all 0
            m_vec.back() = (HEADER0 + 2);
        }
        // the last word is a 0-fill
        else if (m_vec.back() >= HEADER0 && m_vec.back() < FULLBIT0) {
            ++m_vec.back();
        }
        else {
            m_vec.push_back(active.val);
        }
    }
    else if (active.val == ALLONES) {// incoming word is allones
        if (m_vec.back() == ALLONES) {
            m_vec.back() = (HEADER1 | 2);
        }
        // the last word is a 1-fill
        else if (m_vec.back() >= HEADER1 && m_vec.back() < FULLBIT1) {
            ++m_vec.back();
        }
        else {
            m_vec.push_back(active.val);
        }
    }
    else { // incoming word contains a mixture of bits
        m_vec.push_back(active.val);
    }

    for (int lo = 0; lo < SECONDBIT; lo++) {
        if ((active.val >> lo) % 2 == 1) {
            nset += 1;
        }
        else
            continue;
    }
    nbits += MAXBITS;
    active.reset();      
} // bitvector::append_active

///append cnt val-fills 
void ibis::bitvector8::append_fill(int val, word_t cnt) {
    if (cnt > MAXCNT)
        return;
    word_t head = 2 + val;                    /// val is mark the fill number
    word_t w = (head << SECONDBIT) + cnt;     /// w is a fill-word filled with 0 or 1
    if (m_vec.empty()) {
        m_vec.push_back(w);
    }
    else if ((val == 0 && m_vec.back() == FULLBIT0) || (val == 1 && m_vec.back() == FULLBIT1)) {
        m_vec.push_back(w);
    }
    else if (val == 0 && (m_vec.back() >> SECONDBIT) == head) {
        if (m_vec.back() <= FULLBIT0 - cnt)
            m_vec.back() += cnt;
        else {
            m_vec.back() += (FULLBIT0 - m_vec.back());
            w -= (FULLBIT0 - m_vec.back());
            m_vec.push_back(w);
        }
    }
    else if (val == 1 && (m_vec.back() >> SECONDBIT) == head && m_vec.back() <= FULLBIT1 - cnt) {
        if (m_vec.back() <= FULLBIT1 - cnt)
            m_vec.back() += cnt;
        else {
            m_vec.back() += (FULLBIT1 - m_vec.back());
            w -= (FULLBIT1 - m_vec.back());
            m_vec.push_back(w);
        }
    }
    else if ((m_vec.back() == ALLONES) && head == 3 && m_vec.back() <= FULLBIT1 - 1) {
        m_vec.back() = w + 1;
    }
    else if ((m_vec.back() == 0) && head == 2 && m_vec.back() <= FULLBIT0 - 1) {
        m_vec.back() = w + 1;
    }
    else {
        m_vec.push_back(w);
    }
    nbits += cnt * MAXBITS;
    nset += (val > 0) ? cnt * MAXBITS : 0;
} // bitvector::append_fill

// append n val-bits (okay)
void ibis::bitvector8::append_bits(int val, uint64_t n) {
    if (n == 0) return;
    if (active.nbits > 0) {
        word_t tmp = (MAXBITS - active.nbits);
        if (tmp > n) tmp = n;
        active.nbits += tmp;
        active.val <<= tmp;
        n -= tmp;
        if (val != 0)
            active.val |= ((word_t)1 << tmp) - 1;
        if (active.nbits >= MAXBITS)
            append_active();
    }
    if (n >= MAXBITS) {
        uint64_t cnt = n / MAXBITS;
        while (cnt >= MAXCNT) {
            append_fill(val, MAXCNT);
            cnt -= MAXCNT;
        }
        if (cnt > 1)
            append_fill(val, cnt);
        else if (val != 0) {
            active.val = ALLONES;
            append_active();
        }
        else {
            active.val = 0;
            append_active();
        }
        n -= cnt * MAXBITS;
    }
    if (n > 0) {
        active.nbits = n;
        active.val = val * (((word_t)1 << n) - 1);
    }
}

/// add 1s to nv, and add 0s to nt
void ibis::bitvector8::adjustSize(uint64_t nv, uint64_t nt) {
    if (nbits == 0 || nbits < m_vec.size() * MAXBITS)
        nbits = do_cnt();
    return;
    const uint64_t sz = nbits + active.nbits;
    if (sz == nt)
        return;

#if DEBUG + 0 > 0 || _DEBUG + 0 > 0
    LOGGER(ibis::gVerbose > 5)
        << "DEBUG -- bitvector::adjustSize(" << nv << ", " << nt
        << ") on bitvector with size " << sz;
#endif
    if (nt > sz) {     // add some bits to the end
        if (nv > nt)
            nv = nt;
        if (nv > sz) {
            append_bits(1, nv - sz);
            if (nt > nv)
                append_bits(0, nt - nv);
        }
        else {
            append_bits(0, nt - sz);
        }
    }
    else { // truncate cut the extra part
        return;
    }
} // ibis::bitvector8::adjustSize


/// count nbits and nset(OKAY) 
uint64_t ibis::bitvector8::do_cnt() const throw() {
    uint64_t ns = 0;
    uint64_t nb = 0;
    if (m_vec.empty() == 0) {
        for (std::vector<word_t>::const_iterator it = m_vec.begin();
            it < m_vec.end(); ++it) {
            if ((*it) < HEADER0) {     /// literal 
                nb += MAXBITS;
                ns += cnt_ones(*it);
            }
            else {                     /// fill
                uint64_t tmp = (*it & MAXCNT) * MAXBITS;
                nb += tmp;
                ns += tmp * ((*it) >= HEADER1);
            }
        }
        // when nset == 0, this function is invoked again to recompute
        // nset, the following statements make the future computerations faster.
        if (ns == 0 && m_vec.size() > 1) {
            const_cast<word_t&>(m_vec.front()) = (HEADER0 + (ns / MAXBITS));
            const_cast<std::vector<word_t>*>(&m_vec)->resize(1);
        }
    }
    nset = ns;
    return nb;
} // ibis::bitvector::do_cnt

/// count the number of 1(OKAY)
uint64_t ibis::bitvector8::cnt_ones(word_t val) const {
    // number of 1 bits in a value between 0 and 255
    static const word_t table[256] = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    };
    return table[val & 0xFFU] ;
}

/// append a bit(OKAY)
ibis::bitvector8& ibis::bitvector8::operator+=(int b) {
    active.append(b);
    if (active.is_full())
        append_active();
    return *this;
}


void ibis::bitvector8::write(const char* fn) const {
    if (fn == 0 || *fn == 0) return;

    FILE* out = fopen(fn, "wb");
    if (out == 0) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::write failed to open \""
            << fn << "\" to write the bit vector ... "
            << (errno ? strerror(errno) : "no free stdio stream");
        throw "bitvector::write failed to open file";
    }

    int ierr;
    IBIS_BLOCK_GUARD(fclose, out);

    ierr = fwrite((const void*)&(*m_vec.begin()), sizeof(word_t), m_vec.size(),
        out);
    if (ierr != (long)m_vec.size()) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::write only wrote " << ierr
            << " out of " << m_vec.size() << " words to " << fn;
        throw "bitvector::write failed to write all bytes";
    }
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    active_word tmp(active);
    if (active.nbits >= MAXBITS) {
        tmp.nbits = active.nbits % MAXBITS;
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::write detects larger than "
            << "expected active.nbits (" << active.nbits << ", MAX="
            << MAXBITS << "), setting it to " << tmp.nbits;
    }
    const word_t avmax = (1 << tmp.nbits) - 1;
    if (active.val > avmax) {
        tmp.val &= avmax;
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::write detects larger than "
            << "expected active.val (" << active.val << ", MAX=" << avmax
            << "), setting it to " << tmp.val;
    }
    if (tmp.nbits > 0) {
        ierr = fwrite((const void*)&(tmp.val), sizeof(word_t), 1, out);
        LOGGER(ierr < 1 && ibis::gVerbose > 0)
            << "Warning -- bitvector::write failed to write the active word ("
            << tmp.val << ") to " << fn;
    }
    ierr = fwrite((const void*)&(tmp.nbits), sizeof(word_t), 1, out);
    LOGGER(ierr < 1 && ibis::gVerbose > 0)
        << "Warning -- bitvector::write failed to write the number of bits "
        "in the active word (" << tmp.nbits << ") to " << fn;
#else
    if (active.nbits > 0) {
        ierr = fwrite((const void*)&(active.val), sizeof(word_t), 1, out);
        LOGGER(ierr < 1 && ibis::gVerbose > 0)
            << "Warning -- bitvector::write failed to write the active word ("
            << active.val << ") to " << fn;
    }
    
    ierr = fwrite((const void*)&(active.nbits), sizeof(word_t), 1, out);
    LOGGER(ierr < 1 && ibis::gVerbose > 0)
        << "Warning -- bitvector::write failed to write the number of bits "
        "in the active word (" << active.nbits << ") to " << fn;
#endif
} // ibis::bitvector::write


void ibis::bitvector8::read(const char* fn) {

    if (fn == 0 || *fn == 0) return;
    // let the file manager handle the read operation to avoid extra copying
    //ibis::bitvector8 *btv;
    
    int start, end;
    std::ifstream in(fn, std::ios::in | std::ios::binary);
    if(!in)
        std::cout << "fail to find the file" << std::endl;
        
    start = in.tellg();
    in.seekg(0, std::ios::end);
    end = in.tellg();
    
    uint64_t len = end - start ; 
    // std::cout << "len :" << len << std::endl;
    
    std::vector<word_t> img(len);
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(img.data()), len);
    for(int i = 0; i < img.size() - 2; i++){
        //std::cout << bitset<8>(img[i]) << std::endl;
        m_vec.push_back(img[i]);
        //this->m_vec[i] = img[i];
        //std::cout << bitset<8>(m_vec[i]) << std::endl;
    }   
    //std::cout << bitset<8>(img[img.size() - 2]) << std::endl;
    //std::cout << bitset<8>(img.back()) << std::endl << std::endl; 
    //std::cout << "img.size:" << img.size() << std::endl;
    //std::cout << "m_vec.size:" << m_vec.size() << std::endl;
    
    if(m_vec.size() > 1){
        //std::cout << this->m_vec.size() << std::endl << std::endl;
        if(img.back() > 0){
            active.val = img[img.size() - 2];
            active.nbits = img.back();
        }
        else{
            active.val = 0;
            active.nbits = 0;
        }
    }
    nbits = do_cnt();
    img.clear();
    /*
    for(std::vector<word_t>::iterator it = m_vec.begin(); it != m_vec.end(); it++){
        std::cout << bitset<8>(*it) << std::endl;
    }
    std::cout << bitset<8>(active.val) << std::endl << std::endl;*/
    
}
    
    
std::vector<uint8_t> ibis::bitvector8::decode(std::vector<uint8_t> &append_rowids, Table_config *config) {
    std::vector<uint8_t> rids(m_vec.size());

    uint64_t jnd = 0;
    std::vector<word_t>::const_iterator it = m_vec.begin();
    while (it < m_vec.end()) {
        if (*it > HEADER0) { // a fill
            const uint64_t cnt = ((*it) & MAXCNT) * MAXBITS;
            if (*it >= HEADER1) { // 1-fill
                for (uint64_t i = 0; i < cnt; ++i) {
                    if (jnd + i > config->n_rows && append_rowids.size() > 0) {
                        rids.push_back(append_rowids[jnd + i - config->n_rows]);
                    } else {
                        rids.push_back(jnd + i);
                    }
                }
            }
            jnd += cnt;
        }
        else {
            for (int i = MAXBITS; i >= 0; --i) {
                if ((*it >> i) & (word_t)1) {
                    if (jnd + MAXBITS - i > config->n_rows && append_rowids.size() > 0) {
                        rids.push_back(append_rowids[jnd + MAXBITS - i - config->n_rows]);
                    } else {
                        rids.push_back(jnd + MAXBITS - i);
                    }
                }
            }
            jnd += MAXBITS;
        }
        ++it;
    }
    return rids;
}    

//
void ibis::bitvector8::buildIndex() {
    int nWords = 0;
    run8 it;
    it.it = m_vec.begin();
    while (it.it < m_vec.end()) {
        it.decode();
        nWords += it.nWords;
        if (nWords > INDEX_WORDS_2) {
            index.push_back(std::make_pair(it.vecPos, it.wordPos));
            nWords = 0;
        }
        ++it.it;
    }
}

void ibis::bitvector8::compress() {
    if (m_vec.size() < 2) return;

    std::vector<word_t> tmp;
    tmp.reserve(static_cast<uint32_t>(m_vec.size() * 0.382));
    std::vector<word_t>::iterator i0 = m_vec.begin();

    tmp.push_back(*i0);
    for (i0++; i0 != m_vec.end(); i0++) {
        if (*i0 > ALLONES) { // a counter
            if ((*i0 & HEADER1) == (tmp.back() & HEADER1)) { // the same fillBit
                int fillBit = (*i0 > ALLONES);
                word_t nWords = *i0 & MAXCNT;
                word_t tmp_nWords = tmp.back() & MAXCNT;
                if(nWords + tmp_nWords <= MAXCNT)
                    tmp.back() += (*i0 & MAXCNT);
                else {
                    if (fillBit) {
                        *i0 -= (FULLBIT0 - tmp.back());
                        tmp.back() = FULLBIT0;

                        if (*i0 == HEADER0) continue;
                        else if (*i0 == HEADER0 + 1) tmp.push_back(0);
                        else tmp.push_back(*i0);
                    }
                    else {
                        *i0 -= (FULLBIT1 - tmp.back());
                        tmp.back() = FULLBIT1;

                        if (*i0 == HEADER1) continue;
                        else if (*i0 == HEADER1 + 1) tmp.push_back(ALLONES);
                        else tmp.push_back(*i0);
                    }
                }
            } 
            else
                tmp.push_back(*i0);
        }
        else if (*i0 == 0) {
            if (tmp.back() == 0)
                tmp.back() = (HEADER0 | 2);
            else if (tmp.back() > HEADER0 && tmp.back() < FULLBIT0)
                ++tmp.back();
            else
                tmp.push_back(0);
        }
        else if (*i0 == ALLONES) {
            if (tmp.back() == ALLONES)
                tmp.back() = (HEADER1 | 2);
            else if (tmp.back() > HEADER1 && tmp.back() < FULLBIT1)
                ++tmp.back();
            else
                tmp.push_back(ALLONES);
        }
        else {
            tmp.push_back(*i0);
        }
    }

    if (m_vec.size() != tmp.size())
        m_vec.swap(tmp);  // take on the new vector
}


//void ibis::bitvector8::compress() {
//    if (m_vec.size() < 2) // there is nothing to do
//        return;
//
//    struct xrun {
//        bool isFill;
//        int fillBit;
//        int nbits;
//        word_t nWords;
//        std::vector<word_t>::iterator it;
//
//        xrun() : isFill(false), fillBit(0), nWords(0), nbits(0), it(0) { };
//
//        void decode() {
//            fillBit = (*it > HEADER1);
//            isFill = (*it > ALLONES);
//            nWords = (*it & MAXCNT);
//            if(isFill) nbits = MAXBITS * nWords;
//            else nbits = 7;
//        }
//    };
//    xrun last;      // point to the last code word in m_vec that might be modified
//    // NOTE: last.nWords is not used by this function
//    xrun current;   // point to the current code to be examined
//
//    current.it = m_vec.begin() + 1;
//    last.it = m_vec.begin();
//    last.decode();
//    for (; current.it != m_vec.end(); ++current.it) {
//        current.decode();
//        
//        if (*(last.it) == FULLBIT0 || *(last.it) == FULLBIT1) {
//            ++last.it;
//            *(last.it) = *(current.it);
//            last.nbits = current.nbits;
//            last.nWords = current.nWords;
//            continue;          
//        }
//        
//        if(!last.nbits){
//            *(last.it) = *(current.it);
//            last.nbits = current.nbits;
//            last.nWords = current.nWords;
//            continue;
//        }
//        
//        if (last.isFill) { // last word was a fill word
//            /// if last is a fullbit0/1   
//            if (current.isFill) { // current word is a fill word
//                if (current.fillBit == last.fillBit) { // same type of fill
//                    if (current.nWords + last.nWords <= MAXCNT) {
//                        *(last.it) += current.nWords;
//                        last.nbits += 7 * current.nWords;
//                        last.nWords = current.nWords;
//                    }
//                    else {   
//                        if (!current.fillBit){
//                            *(current.it) -= (FULLBIT0 - *(last.it));
//                            current.nWords -= (FULLBIT0 - *(last.it));
//                            *(last.it) = FULLBIT0;
//                        }
//                        else{
//                            *(current.it) -= (FULLBIT1 - *(last.it));
//                            current.nWords -= (FULLBIT1 - *(last.it));
//                            *(last.it) = FULLBIT1;                            
//                        }    
//                                           
//                        ++last.it;
//                        last.nbits = 0;
//                        
//                        if (current.nWords >= 2) {
//                            *(last.it) = *(current.it);
//                            last.isFill = true;
//                            last.fillBit = current.fillBit;
//                            last.nWords = current.nWords;
//                            last.nbits = 7 * current.nWords;
//                        }
//                        else if(current.nWords) {
//                            if(!current.fillBit) *(last.it) = 0;
//                            else *(last.it) = ALLONES;
//                            last.isFill = false;
//                            last.nbits = 7;
//                        }
//                        else{
//                            *(last.it) = 0;
//                        }
//                    }
//                }
//                else { // different types of fills, move last foward by one
//                    ++last.it;
//                    *(last.it) = *(current.it);
//                    last.fillBit = current.fillBit;
//                    last.nWords = current.nWords;
//                    last.nbits = 7;
//                }
//            }
//            else if (last.fillBit == 0 && *(current.it) == 0){ 
//                // increase the last fill by 1 word
//                if(*(last.it) < FULLBIT0){
//                    ++*(last.it);
//                    ++last.nWords;
//                    last.nbits += 7;
//                }
//            }
//            else if (last.fillBit == 1 && *(current.it) == ALLONES){ 
//                if(*(last.it) < FULLBIT1){
//                    ++*(last.it);
//                    ++last.nWords;
//                    last.nbits += 7;
//                }       
//                /*else{
//                    *(last.it) = FULLBIT1;
//                    last.it++;
//                    *(last.it) = ALLONES;
//                    last.isFill = false;
//                }*/
//            }
//            else { // move last forward by one
//                ++last.it;
//                last.isFill = false;
//                *(last.it) = *(current.it);
//                last.nWords = current.nWords;
//                last.nbits = 7;
//            }
//        }
//        else if (current.isFill && last.nbits != 0) {
//            // last word was a literal word, current word is a fill word
//            if (current.fillBit == 0 && *(last.it) == 0){
//                // change the last word into a fill word
//                if(current.nWords < MAXCNT) {
//                    *(last.it) = *(current.it) + 1;
//                    last.fillBit = current.fillBit;
//                    last.isFill = true;
//                    last.nWords = current.nWords + 1;
//                    last.nbits = 7 * (current.nWords + 1);
//                }
//                /*else {
//                    *(last.it) = FULLBIT0;
//                    ++last.it;
//                    *(last.it) = 0;
//                    last.nbits = 7;
//                    last.isFill = false;
//                }*/
//            }
//            else if (current.fillBit == 1 && *(last.it) == ALLONES){
//                if(current.nWords < MAXCNT) {
//                    *(last.it) = *(current.it) + 1;
//                    last.fillBit = current.fillBit;
//                    last.isFill = true;
//                    last.nWords = current.nWords + 1;
//                    last.nbits = 7 * (current.nWords + 1);
//                }
//                /*else {
//                    *(last.it) = FULLBIT1;
//                    ++last.it;
//                    *(last.it) = ALLONES;
//                    last.isFill = false;
//                }*/
//            }
//            else { // move last forward by one
//                ++last.it;
//                last.isFill = true;
//                *(last.it) = *(current.it);
//                last.fillBit = current.fillBit;
//                last.nbits = current.nbits;
//                last.nWords = current.nWords;
//            }
//        }
//        else if (*(last.it) == *(current.it) && last.nbits != 0) {
//            // both last word and current word are literal words and are the same
//            if (*(current.it) == 0) { // make a 2-word 0-fill
//                *(last.it) = (HEADER0 | 2);
//                last.isFill = true;
//                last.fillBit = 0;
//                last.nbits = 14;
//                last.nWords = 2;
//            }
//            else if (*(current.it) == ALLONES) { // make a 2-word 1-fill
//                *(last.it) = (HEADER1 | 2);   
//                last.isFill = true;
//                last.fillBit = 1;
//                last.nbits = 14;
//                last.nWords = 2;
//            }
//            else { // move last forward
//                ++last.it;
//                *(last.it) = *(current.it);
//                last.nWords = current.nWords;
//                last.nbits = 7;
//            }
//        }
//        else { // move last forward one word
//            ++last.it;
//            *(last.it) = *(current.it);
//            last.nWords = current.nWords;
//            last.nbits = current.nbits;
//        }
//    }
//    
//    ++last.it;
//    if (last.it < m_vec.end()) { // reduce the size of m_vec
//        m_vec.erase(last.it, m_vec.end());
//    }
//}

void ibis::bitvector8::decompress() {
    if (nbits == 0 && m_vec.size() > 0){     
        nbits = do_cnt();
    }
    // std::cout << "btv8 nbits: " << nbits <<std::endl;
    if (m_vec.size() * MAXBITS == nbits) {// already uncompressed
         std::cout << "the btv8 has been decompressed !" << std::endl;
        return;
    }
    std::vector<word_t> tmp;
    tmp.resize(nbits / MAXBITS);
    if (nbits != tmp.size() * MAXBITS) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::decompress(nbits=" << nbits
            << ") failed to allocate a temp array of "
            << nbits / MAXBITS << "-word";
        throw ibis::bad_alloc("bitvector::decompress failed to "
            "allocate array to uncompressed bits");
    }

    std::vector<word_t>::iterator it = tmp.begin();
    std::vector<word_t>::const_iterator i0 = m_vec.begin();
    for (; i0 != m_vec.end(); i0++) {
        if ((*i0) > ALLONES) {        /// fill
            word_t cnt = (*i0 & MAXCNT);
            if ((*i0) >= HEADER1) {
                for (word_t j = 0; j < cnt; j++, it++)
                    *it = ALLONES;
            }
            else {
                for (word_t j = 0; j < cnt; j++, it++)
                    *it = 0;
            }
        }
        else {   // literal
            *it = *i0;
            it++;
        }
    }

    if (m_vec.size() != tmp.size())
        m_vec.swap(tmp);  // take on the new vector
}

/// Decompress the current content to an vector<word_t>.
void ibis::bitvector8::decompress(std::vector<ibis::bitvector8::word_t>& tmp)
const {
    const uint64_t nb = ((nbits == 0 && m_vec.size() > 0) ? do_cnt() : nbits);
    uint64_t cnt = nb / MAXBITS;
    tmp.resize(cnt);

    std::vector<word_t>::iterator it = tmp.begin();
    std::vector<word_t>::const_iterator i0 = m_vec.begin();
    for (; i0 != m_vec.end(); i0++) {
        if ((*i0) > ALLONES) {
            cnt = (*i0 & MAXCNT);
            if ((*i0) >= HEADER1) {
                for (uint64_t j = 0; j < cnt; j++, it++)
                    *it = ALLONES;
            }
            else {
                for (uint64_t j = 0; j < cnt; j++, it++)
                    *it = 0;
            }
        }
        else {
            *it = *i0;
            it++;
        }
    }
} // ibis::bitvector8::decompress

void ibis::bitvector8::countWords(){
    int num_fill = 0;
    int num_lite = 0;
    int tmp = 0;
    if (m_vec.empty() == 0) 
        for (std::vector<word_t>::const_iterator it = m_vec.begin();
            it < m_vec.end(); ++it) {
            if ((*it) < HEADER0) {     /// literal 
                num_lite++;
                tmp++;
            }
            else {                     /// fill
                num_fill++;
                tmp += (*it & MAXCNT);
            }
        }
    std::cout << "number of 7-block : " << tmp
              << " number of fill-word: " << num_fill
              << " number of literal-word: " << num_lite << std::endl;
}


/// Decompress the current content to an vector<word_t> and complement
/// every bit.
void ibis::bitvector8::copy_comp(std::vector<ibis::bitvector8::word_t>& tmp) const {
    uint64_t cnt = (nbits == 0 && m_vec.size() > 0 ? do_cnt() : nbits) / MAXBITS;
    tmp.resize(cnt);

    std::vector<word_t>::iterator it = tmp.begin();
    std::vector<word_t>::const_iterator i0 = m_vec.begin();
    for (; i0 != m_vec.end(); i0++) {
        if ((*i0) > ALLONES) {
            word_t _cnt = (*i0 & MAXCNT);
            if ((*i0) >= HEADER1) {
                for (word_t j = 0; j < _cnt; j++, it++)
                    *it = 0;
            }
            else {
                for (word_t j = 0; j < _cnt; j++, it++)
                    *it = ALLONES;
            }
        }
        else {
            *it = ALLONES ^ *i0;
            it++;
        }
    }
} // ibis::bitvector8::copy_comp


/// The in-place version of the bitwise logical AND operator.  It performs
/// the bitwise logical AND operation between this bitvector and  rhs,
/// then stores the result back to this bitvector.
///
///@note If the two bit vectors are not of the same length, the shorter one
/// is implicitly padded with 0 bits so the two are of the same length.
void ibis::bitvector8::operator&=(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure the memory is not shared
    if (size() > rhs.size()) { // make a copy of RHS and extend its size
        ibis::bitvector8 tmp(rhs);
        tmp.adjustSize(0, size());
        operator&=(tmp);
        return;
    }
    else if (size() < rhs.size()) {
        adjustSize(0, rhs.size());
    }

    const bool ca = (m_vec.size() * MAXBITS == nbits && nbits > 0);
    const bool cb = (rhs.m_vec.size() * MAXBITS == rhs.nbits && rhs.nbits > 0);
    if (ca) {   // *this is not compressed
        if (cb) // rhs is not compressed
            and_c0(rhs);
        else
            and_d1(rhs);
    }
    else if (cb) { // rhs is not compressed
        bitvector8 tmp;
        tmp.copy(rhs);
        swap(tmp);
        and_d1(tmp);
    }
    else if (all0s() || rhs.all1s()) { // deal with active words
        if (active.nbits == rhs.active.nbits) {
            active.val &= rhs.active.val;
        }
        else if (active.nbits > rhs.active.nbits) {
            active.val &= (rhs.active.val << (active.nbits - rhs.active.nbits));
        }
        else {
            active.val = (active.val << (rhs.active.nbits - active.nbits))
                & rhs.active.val;
            active.nbits = rhs.active.nbits;
        }
    }
    else if (rhs.all0s() || all1s()) { // copy rhs
        nset = rhs.nset;
        m_vec = rhs.m_vec;
        if (active.nbits == rhs.active.nbits) {
            active.val &= rhs.active.val;
        }
        else if (active.nbits > rhs.active.nbits) {
            active.val &= (rhs.active.val << (active.nbits - rhs.active.nbits));
        }
        else {
            active.val = (active.val << (rhs.active.nbits - active.nbits))
                & rhs.active.val;
            active.nbits = rhs.active.nbits;
        }
    }
    else if ((m_vec.size() + rhs.m_vec.size()) * MAXBITS >= rhs.nbits) {
        // if the total size of the two operands are large, generate an
        // decompressed solution
        bitvector8 res;
        and_d2(rhs, res);
        swap(res);
    }
    else {
        bitvector8 res;
        and_c2(rhs, res);
        swap(res);
    }
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    uint64_t nb = do_cnt();
    if (nbits == 0)
        nbits = nb;
    LOGGER(nb != rhs.nbits && rhs.nbits > 0 && ibis::gVerbose > 0)
        << "Warning -- bitvector::operator&= expects to have " << rhs.nbits
        << " bits but got " << nb;
    LOGGER(ibis::gVerbose > 30 || ((1U << ibis::gVerbose) >= bytes() &&
        (1U << ibis::gVerbose) >= rhs.bytes()))
        << "operator&=: (A&B)=" << *this;
#endif
} // ibis::bitvector8::operator&=

/// The is the in-place version of the bitwise OR (|) operator.  This
/// bitvector8 is modified to store the result of the operation.
void ibis::bitvector8::operator|=(const ibis::bitvector8& rhs) {
#if defined(WAH_CHECK_SIZE)
    if (nbits == 0)
        nbits = do_cnt();
    LOGGER((rhs.nbits > 0 && nbits != rhs.nbits) ||
        active.nbits != rhs.active.nbits)
        << "Warning -- bitvector::operator|= is to operate on two bitvectors "
        "of different sizes (" << size() << " != " << rhs.size() << ')';
#endif
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    LOGGER(ibis::gVerbose > 30 || ((1U << ibis::gVerbose) >= bytes() &&
        (1U << ibis::gVerbose) >= rhs.bytes()))
        << "operator|=: A=" << *this << "B=" << rhs;
#endif
    // m_vec.nosharing();
    if (size() > rhs.size()) {
        ibis::bitvector8 tmp(rhs);
        tmp.adjustSize(0, size());
        operator|=(tmp);
        return;
    }
    else if (size() < rhs.size()) {
        adjustSize(0, rhs.size());
    }

    const bool ca = (m_vec.size() * MAXBITS == nbits && nbits > 0);
    const bool cb = (rhs.m_vec.size() * MAXBITS == rhs.nbits && rhs.nbits > 0);
    if (ca) {        // *this is not compressed
        if (cb)      // rhs is not compressed
            or_c0(rhs);
        else
            or_d1(rhs);
    }
    else if (cb) {
        bitvector8 tmp;
        tmp.copy(rhs);
        swap(tmp);
        or_d1(tmp);
    }
    else if (all1s() || rhs.all0s()) {
        active.val |= rhs.active.val;
    }
    else if (all0s() || rhs.all1s()) {
        nset = rhs.nset;
        m_vec = rhs.m_vec;
        active.val |= rhs.active.val;
    }
    else if ((m_vec.size() + rhs.m_vec.size()) * MAXBITS >= rhs.nbits) {
        // if the total size of the two operands are large, generate an
        // uncompressed result
        bitvector8 res;
        or_d2(rhs, res);
        swap(res);
    }
    else {
        bitvector8 res;
        or_c2(rhs, res);
        swap(res);
    }
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    word_t nb = do_cnt();
    if (nbits == 0)
        nbits = nb;
    LOGGER(nb != rhs.nbits && rhs.nbits > 0)
        << "Warning -- bitvector::operator|= expects to have " << rhs.nbits
        << " bits but got " << nb;
    LOGGER(ibis::gVerbose > 30 || ((1U << ibis::gVerbose) >= bytes() &&
        (1U << ibis::gVerbose) >= rhs.bytes()))
        << "operator|=: (A|B)=" << *this;
#endif
} // ibis::bitvector8::operator|=

/// The in-place version of the bitwise XOR (^) operator.  This bitvector
/// is modified to store the result.
///
///@sa ibis::bitvector8::operator^=
void ibis::bitvector8::operator^=(const ibis::bitvector8& rhs) {
    // m_vec.nosharing();
    if (size() > rhs.size()) {
        ibis::bitvector8 tmp(rhs);
        tmp.adjustSize(0, size());
        operator^=(tmp);
        return;
    }
    else if (size() < rhs.size()) {
        adjustSize(0, rhs.size());
    }
 
    const bool ca = (m_vec.size() * MAXBITS == nbits && nbits > 0);
    const bool cb = (rhs.m_vec.size() * MAXBITS == rhs.nbits && rhs.nbits > 0);
    if (ca) {
        if (cb)
            xor_c0(rhs);
        else
            xor_d1(rhs);
    }
    else if (cb) {
        bitvector8 res;
        xor_c1(rhs, res);
        swap(res);
    }
    else if ((m_vec.size() + rhs.m_vec.size()) * MAXBITS >= rhs.nbits) {
        // if the total size of the two operands are large, generate an
        // uncompressed result
        // std::cout << "xor_d2" << std::endl;
        bitvector8 res;
        xor_d2(rhs, res);
        swap(res);
    }
    else {
        // std::cout << "xor_c2" << std::endl;
        bitvector8 res;
        xor_c2(rhs, res);
        swap(res);
    }
} // ibis::bitvector8::operator^=

// both operands of the 'or' operation are not compressed
void ibis::bitvector8::or_c0(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure *this is not shared!
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "operand 1 of OR" << *this << std::endl;
        lg() << "operand 2 of OR" << rhs;
    }
#endif
    nset = 0;
    std::vector<word_t>::iterator i0 = m_vec.begin();
    std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
    while (i0 != m_vec.end()) {    // go through all words in m_vec
        *i0 |= *i1;
        i0++;
        i1++;
    } // while (i0 != m_vec.end())

    // the last thing -- work with the two active_words
    active.val |= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    word_t nb = do_cnt();
    if (nbits == 0)
        nbits = nb;
    LOGGER(nb != rhs.nbits && rhs.nbits > 0)
        << "Warning -- bitvector::or_c0= expects to have " << rhs.nbits
        << " bits but got " << nb;
    LOGGER(ibis::gVerbose > 4) << "result of OR " << *this;
#endif
} // ibis::bitvector8::or_c0

// or operation where rhs is not compressed (not one word is a counter)
void ibis::bitvector8::or_c1(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    res.clear();
    if (m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = m_vec;
            res.nbits = nbits;
            res.nset = nbits;
        }
        else if (*it > ALLONES) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = rhs.nset;
        }
        else {
            res.m_vec.push_back(*it | *(rhs.m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (m_vec.size() > 1) {
        std::vector<word_t>::const_iterator i0 = m_vec.begin();
        std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin(), i2;
        word_t s0;
        res.m_vec.reserve(rhs.m_vec.size());
        while (i0 != m_vec.end()) {    // go through all words in m_vec
            if (*i0 > ALLONES) { // i0 is compressed
                s0 = ((*i0) & MAXCNT);
                if ((*i0) >= HEADER1) { // the result is all ones
                    if (s0 > 1)
                        res.append_fill(1, s0);
                    else {
                        res.active.val = ALLONES;
                        res.append_active();
                    }
                    i1 += s0;
                }
                else { // the result is *i1
                    i2 = i1 + s0;
                    for (; i1 < i2; i1++)
                        res.m_vec.push_back(*i1);
                    res.nbits += s0 * MAXBITS;
                }
            }
            else { // both words are not compressed
                res.active.val = *i0 | *i1;
                res.append_active();
                i1++;
            }
            i0++;
        } // while (i0 != m_vec.end())

        if (i1 != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_c1 expects to exhaust i1 but "
                "there are " << (rhs.m_vec.end() - i1) << " word(s) left";
            throw "or_c1 internal error";
        }
    }

    // the last thing -- work with the two active_words
    res.active.val = active.val | rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "operand 1 of OR" << *this << std::endl;
    lg() << "operand 2 of OR" << rhs << std::endl;
    lg() << "result or OR " << res;
#endif
} // ibis::bitvector8::or_c1

// or operation on two compressed bitvectors
void ibis::bitvector8::or_c2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    res.clear();
    if (m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = m_vec;
            res.nbits = nbits;
            res.nset = nbits;
        }
        else if (*it > ALLONES) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = rhs.nset;
        }
        else {
            res.m_vec.push_back(*it | *(rhs.m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (rhs.m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = rhs.nbits;
        }
        else if (*it > ALLONES) {
            res.m_vec= m_vec;
            res.nbits = nbits;
            res.nset = nset;
        }
        else {
            res.m_vec.push_back(*it | *(m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (m_vec.size() > 1) {
        run8 x, y;
        x.it = m_vec.begin();
        y.it = rhs.m_vec.begin();
        while (x.it != m_vec.end()) {    // go through all words in m_vec
            if (x.nWords == 0)
                x.decode();
            if (y.nWords == 0)
                y.decode();
#if defined(_DEBUG) || defined(DEBUG)
            LOGGER((x.nWords == 0 || y.nWords == 0) && ibis::gVerbose >= 0)
                << " Error -- bitvector::or_c2 serious problem here ...";
#endif
            if (x.isFill != 0) { // x points to a fill
                // if both x and y point to fills, use the longer one
                if (y.isFill != 0) {
                    if (y.fillBit != 0) {
                        res.append_fill(y.fillBit, y.nWords);
                        x -= y.nWords;
                        y.nWords = 0;
                        ++y.it;
                    }
                    else if (y.nWords < x.nWords) {
                        res.append_fill(x.fillBit, y.nWords);
                        x.nWords -= y.nWords;
                        y.nWords = 0;
                        ++y.it;
                    }
                    else {
                        res.copy_runs(x, y.nWords);
                        y.it += (y.nWords == 0);
                    }
                }
                else if (x.fillBit) { // the result is all ones
                    res.append_fill(x.fillBit, x.nWords);
                    y -= x.nWords; // advance the pointer in y
                    x.nWords = 0;
                    ++x.it;
                }
                else { // copy the content of y
                    res.copy_runs(y, x.nWords);
                    x.it += (x.nWords == 0);
                }
            }
            else if (y.isFill != 0) { // y points to a fill
                if (y.fillBit) {
                    res.append_fill(y.fillBit, y.nWords);
                    x -= y.nWords;
                    y.nWords = 0;
                    ++y.it;
                }
                else {
                    res.copy_runs(x, y.nWords);
                    y.it += (y.nWords == 0);
                }
            }
            else { // both words are not compressed
                res.active.val = *x.it | *y.it;
                res.append_active();
                x.nWords = 0;
                y.nWords = 0;
                ++x.it;
                ++y.it;
            }
        } // while (x.it < m_vec.end())

        if (x.it != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_c2 expects to exhaust i0 "
                "but there are " << (m_vec.end() - x.it) << " word(s) left";
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
            {
                ibis::util::logger lg(4);
                lg() << "*this: " << *this << std::endl;
                lg() << "rhs: " << rhs << std::endl;
                lg() << "res: " << res;
            }
#else
            throw "or_c2 internal error";
#endif
        }

        if (y.it != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_c2 expects to exhaust i1 "
                "but there are " << (rhs.m_vec.end() - y.it) << " word(s) left";
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
            {
                ibis::util::logger lg(4);
                lg() << "*this: " << *this << std::endl;
                lg() << "rhs: " << rhs << std::endl;
                lg() << "res: " << res;
            }
#else
            throw "or_c2 internal error";
#endif
        }
    }

    // work with the two active_words
    res.active.val = active.val | rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "operand 1 of OR" << *this << std::endl;
    lg() << "operand 2 of OR" << rhs << std::endl;
    lg() << "result of OR " << res;
#endif
} // ibis::bitvector8::or_c2

// assuming *this is uncompressed but rhs is not, this function performs
// the OR operation and overwrites *this with the result
void ibis::bitvector8::or_d1(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure *this is not shared!
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "operand 1 of OR" << *this << std::endl;
        lg() << "operand 2 of OR" << rhs;
    }
#endif
    if (rhs.m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) {      // all ones
            rhs.decompress(m_vec);
            nset = nbits;
        }
        else if (*it <= ALLONES) { // one literal word
            m_vec[0] = (*it | *(rhs.m_vec.begin()));
            nset = cnt_ones(m_vec[0]);
        }
    }
    else if (rhs.m_vec.size() > 1) {
        std::vector<word_t>::iterator i0 = m_vec.begin();
        std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
        word_t s0;
        nset = 0;
        while (i1 != rhs.m_vec.end()) {    // go through all words in m_vec
            if (*i1 > ALLONES) {           // i1 is compressed
                s0 = ((*i1) & MAXCNT);
                if ((*i1) >= HEADER1) {    // the result is all ones
                    std::vector<word_t>::const_iterator stp = i0 + s0;
                    while (i0 < stp) {
                        *i0 = ALLONES;
                        ++i0;
                    }
                }
                else { // the result is *i0
                    i0 += s0;
                }
            }
            else {     // both words are not compressed
                *i0 |= *i1;
                ++i0;
            }
            ++i1;
        } // while (i1 != rhs.m_vec.end())

        if (i0 != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_d1 expects to exhaust i0 but "
                "there are " << (m_vec.end() - i0) << " word(s) left";
            throw "or_d1 internal error";
        }
    }

    // the last thing -- work with the two active_words
    active.val |= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "result of OR" << *this;
    }
#endif
} // ibis::bitvector8::or_d1

// or operation on two compressed bitvectors, generates uncompressed result
void ibis::bitvector8::or_d2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    // establish a uncompressed bitvector with the right size
    res.nbits = ((nbits == 0 && m_vec.size() > 0) ?
        (rhs.nbits == 0 && rhs.m_vec.size() > 0) ?
        (m_vec.size() <= rhs.m_vec.size() ? do_cnt() : rhs.do_cnt())
        : rhs.nbits : nbits);
    res.m_vec.resize(res.nbits / MAXBITS);

    // fill res with the right numbers
    if (m_vec.size() == 1) { // only one word in *this
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) { // all ones
            decompress(res.m_vec);
            res.nset = nbits;
        }
        else if (*it > ALLONES) { // copy rhs
            rhs.decompress(res.m_vec);
            res.nset = rhs.nset;
        }
        else {
            res.m_vec[0] = (*it | *(rhs.m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (rhs.m_vec.size() == 1) { // only one word in rhs
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) { // all ones
            rhs.decompress(res.m_vec);
            res.nset = rhs.nbits;
        }
        else if (*it > ALLONES) { // copy *this
            decompress(res.m_vec);
            res.nset = nset;
        }
        else {
            res.m_vec[0] = (*it | *(m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (m_vec.size() > 1) { // more than one word in *this
        run8 x, y;
        res.nset = 0;
        x.it = m_vec.begin();
        y.it = rhs.m_vec.begin();
        std::vector<word_t>::iterator ir = res.m_vec.begin();
        while (x.it < m_vec.end()) {    // go through all words in m_vec
            if (x.nWords == 0)
                x.decode();
            if (y.nWords == 0)
                y.decode();
            if (x.isFill != 0) {
                if (y.isFill != 0 && y.nWords >= x.nWords) {
                    if (y.fillBit == 0) {
                        res.copy_runs(ir, x, y.nWords);
                        y.it += (y.nWords == 0);
                    }
                    else {
                        x -= y.nWords;
                        res.copy_fill(ir, y);
                    }
                }
                else if (x.fillBit == 0) {
                    res.copy_runs(ir, y, x.nWords);
                    x.it += (x.nWords == 0);
                }
                else {
                    y -= x.nWords;
                    res.copy_fill(ir, x);
                }
            }
            else if (y.isFill != 0) {
                if (y.fillBit == 0) {
                    res.copy_runs(ir, x, y.nWords);
                    y.it += (y.nWords == 0);
                }
                else {
                    x -= y.nWords;
                    res.copy_fill(ir, y);
                }
            }
            else {       // both words are not compressed
                *ir = *x.it | *y.it;
                x.nWords = 0;
                y.nWords = 0;
                ++x.it;
                ++y.it;
                ++ir;
            }
        } // while (x.it < m_vec.end())

        /*if (x.it != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_d2 expects to exhaust i0 but "
                "there are " << (m_vec.end() - x.it) << " word(s) left\n"
                << "Operand 1 or or_d2 " << *this << "\nOperand 2 or or_d2 "
                << rhs << "\nResult of or_d2 " << res;
            throw "or_d2 internal error";
        }

        if (y.it != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::or_d2 expects to exhaust i1 but "
                "there are " << (rhs.m_vec.end() - y.it) << " word(s) left\n"
                << "Operand 1 or or_d2 " << *this << "\nOperand 2 or or_d2 "
                << rhs << "\nResult of or_d2 " << res;
            throw "or_d2 internal error";
        }

        if (ir != res.m_vec.end()) {
            LOGGER(ibis::gVerbose >= 0)
                << "Warning -- bitvector::or_d2 expects to exhaust ir but "
                "there are " << (res.m_vec.end() - ir) << " word(s) left\n"
                << "Operand 1 or or_d2 " << *this << "\nOperand 2 or or_d2 "
                << rhs << "\nResult of or_d2 " << res;
            throw "or_d2 internal error";
        }*/
    }

    // work with the two active_words
    res.active.val = active.val | rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    LOGGER(ibis::gVerbose >= 0)
        << "Operand 1 or or_d2 " << *this
        << "\nOperand 2 or or_d2 " << rhs
        << "\nResult of or_d2 " << res;
#endif
} // ibis::bitvector8::or_d2

// both operands of the 'and' operation are not compressed
void ibis::bitvector8::and_c0(const ibis::bitvector8& rhs) {
    nset = 0;
    // m_vec.nosharing(); // make sure *this is not shared!
    std::vector<word_t>::iterator i0 = m_vec.begin();
    std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
    while (i0 != m_vec.end()) {    // go through all words in m_vec
        *i0 &= *i1;
        i0++;
        i1++;
    } // while (i0 != m_vec.end())

    // the last thing -- work with the two active_words
    active.val &= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result of AND " << *this;
#endif
} // ibis::bitvector8::and_c0

// and operation where rhs is not compressed (not one word is a counter)
void ibis::bitvector8::and_c1(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    res.clear();
    if (m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = rhs.nset;
        }
        else if (*it > ALLONES) {
            res.m_vec = m_vec;
            res.nbits = nbits;
            res.nset = 0;
        }
        else {
            res.m_vec.push_back(*it & *(rhs.m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (m_vec.size() > 1) {
        std::vector<word_t>::const_iterator i0 = m_vec.begin();
        std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin(), i2;
        word_t s0;
        res.m_vec.reserve(rhs.m_vec.size());
        while (i0 != m_vec.end()) {    // go through all words in m_vec
            if (*i0 > ALLONES) { // i0 is compressed
                s0 = ((*i0) & MAXCNT);
                if ((*i0) < HEADER1) { // the result is all zero
                    if (s0 > 1)
                        res.append_fill(0, s0);
                    else
                        res.append_active();
                    i1 += s0;
                }
                else { // the result is *i1
                    i2 = i1 + s0;
                    for (; i1 < i2; i1++)
                        res.m_vec.push_back(*i1);
                    res.nbits += s0 * MAXBITS;
                }
            }
            else { // both words are not compressed
                res.active.val = *i0 & *i1;
                res.append_active();
                i1++;
            }
            i0++;
        } // while (i0 != m_vec.end())

        if (i1 != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_c1 expects to exhaust i1 "
                "but there are " << (rhs.m_vec.end() - i1) << " word(s) left";
            throw "and_c1 iternal error";
        }
    }

    // the last thing -- work with the two active_words
    res.active.val = active.val & rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result of AND " << res;
#endif
} // ibis::bitvector8::and_c1

// bitwise and (&) operation -- both operands may contain compressed words
void ibis::bitvector8::and_c2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    int bi_idx_pos = 0;
    res.clear();
    if (m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = rhs.nset;
        }
        else if (*it > ALLONES) {
            res.m_vec = m_vec;
            res.nbits = nbits;
            res.nset = 0;
        }
        else {
            res.m_vec.push_back(*it & *(rhs.m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (rhs.m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) {
            res.m_vec = m_vec;
            res.nbits = nbits;
            res.nset = nset;
        }
        else if (*it > ALLONES) {
            res.m_vec = rhs.m_vec;
            res.nbits = rhs.nbits;
            res.nset = 0;
        }
        else {
            res.m_vec.push_back(*it & *(m_vec.begin()));
            res.nbits = nbits;
        }
    }
    else if (m_vec.size() > 1) {
        run8 x, y;
        x.it = m_vec.begin();
        y.it = rhs.m_vec.begin();
        if (enable_fence_pointer)
            res.m_vec.reserve(m_vec.size() + rhs.m_vec.size());
        while (x.it < m_vec.end()) { // go through all words in m_vec
            if (x.nWords == 0)
                x.decode();
            if (y.nWords == 0)
                y.decode();
            if (x.isFill != 0) {        // x points to a fill
                // if both x and y point to fills, use the long one
                if (y.isFill != 0) {
                    if (y.fillBit == 0) {
                        res.append_fill(0, y.nWords);
                        x -= y.nWords;
                        y.nWords = 0;
                        ++y.it;
                    }
                    else if (y.nWords >= x.nWords) {
                        res.copy_runs(x, y.nWords);
                        y.it += (y.nWords == 0);
                    }
                    else {
                        res.append_fill(x.fillBit, y.nWords);
                        x.nWords -= y.nWords;
                        y.nWords = 0;
                        ++y.it;
                    }
                }
                else if (x.fillBit == 0) { // generate a 0-fill as the result
                    res.append_fill(0, x.nWords);
                    y -= x.nWords;
                    x.nWords = 0;
                    ++x.it;
                }
                else { // copy the content of y
                    if (enable_fence_pointer)
                        res.copy_runs(y, x.nWords, rhs.index, bi_idx_pos);
                    else
                        res.copy_runs(y, x.nWords);
                    x.it += (x.nWords == 0);
                }
            }
            else if (y.isFill != 0) {        // i1 is compressed
                if (y.fillBit == 0) { // generate a 0-fill as the result
                    res.append_fill(0, y.nWords);
                    x -= y.nWords;
                    y.nWords = 0;
                    ++y.it;
                }
                else { // copy the content of x
                    res.copy_runs(x, y.nWords);
                    y.it += (y.nWords == 0);
                }
            }
            else { // both words are not compressed
                res.active.val = *(x.it) & *(y.it);
                res.append_active();
                x.nWords = 0;
                y.nWords = 0;
                ++x.it;
                ++y.it;
            }
        } // while (x.it < m_vec.end())

        if (x.it != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_c2 expects to exhaust i0 "
                "but there are " << (m_vec.end() - x.it) << " word(s) left";
            throw "and_c2 iternal error";
        }

        if (y.it != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_c2 expects to exhaust i1 "
                "but there are " << (rhs.m_vec.end() - y.it) << " word(s) left";
            throw "and_c2 iternal error";
        }
    }

    // the last thing -- work with the two active_words
    if (active.nbits) {
        res.active.val = active.val & rhs.active.val;
        res.active.nbits = active.nbits;
    }
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result of AND " << res;
#endif
} // bitvector& ibis::bitvector::and_c2

// assuming *this is uncompressed but rhs is not, this function performs the
// AND operation and overwrites *this with the result
void ibis::bitvector8::and_d1(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure *this is not shared!
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "operand 1 of AND" << *this << std::endl;
        lg() << "operand 2 of AND" << rhs << std::endl;
    }
#endif
    if (rhs.m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it < HEADER1) { // nothing to do for *it >= HEADER1
            if (*it > ALLONES) { // all zero
                // memset((void*)m_vec.begin(), 0, sizeof(word_t) * m_vec.size());
                nset = 0;
            }
            else { // one literal word
                m_vec[0] = (*it & *(rhs.m_vec.begin()));
                nset = cnt_ones(m_vec[0]);
            }
        }
    }
    else if (rhs.m_vec.size() > 1) {
        std::vector<word_t>::iterator i0 = m_vec.begin();
        std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
        word_t s0;
        nset = 0;
        while (i1 != rhs.m_vec.end()) {    // go through all words in m_vec
            if (*i1 > ALLONES) { // i1 is compressed
                s0 = ((*i1) & MAXCNT);
                if ((*i1) < HEADER1) { // set literal words to zero
                    // memset((void*)i0, 0, sizeof(word_t) * s0);
                }
                i0 += s0;
            }
            else { // both words are not compressed
                *i0 &= *i1;
                ++i0;
            }
            ++i1;
        } // while (i1 != rhs.m_vec.end())

        if (i0 != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_d1 expects to exhaust i0 "
                "but there are " << (m_vec.end() - i0) << " word(s) left";
            throw "and_d1 internal error";
        }
    }

    // the last thing -- work with the two active_words
    active.val &= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "result of AND" << *this;
    }
#endif
} // ibis::bitvector::and_d1

// and operation on two compressed bitvectors, generates uncompressed result
void ibis::bitvector8::and_d2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    LOGGER(ibis::gVerbose > 2)
        << "DEBUG -- bitvector::and_d2 -- starting with \nOperand 1\n"
        << *this << "\nOperand 2\n" << rhs;
#endif
    // establish a uncompressed bitvector with the right size
    res.nbits = ((nbits == 0 && m_vec.size() > 0) ?
        (rhs.nbits == 0 && rhs.m_vec.size() > 0) ?
        (m_vec.size() <= rhs.m_vec.size() ? do_cnt() : rhs.do_cnt())
        : rhs.nbits : nbits);
    res.m_vec.resize(res.nbits / MAXBITS);

    // fill res with the right numbers
    if (m_vec.size() == 1) { // only one word in *this
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) { // copy rhs
            rhs.decompress(res.m_vec);
            res.nset = rhs.nset;
        }
        else if (*it > ALLONES) { // all zeros
            decompress(res.m_vec);
            res.nset = 0;
        }
        else {
            res.m_vec[0] = (*it & *(rhs.m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (rhs.m_vec.size() == 1) { // only one word in rhs
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) { // copy *this
            decompress(res.m_vec);
            res.nset = nset;
        }
        else if (*it > ALLONES) { // all zero
            rhs.decompress(res.m_vec);
            res.nset = 0;
        }
        else {
            res.m_vec[0] = (*it & *(m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (m_vec.size() > 1) { // more than one word in *this
        run8 x, y;
        res.nset = 0;
        x.it = m_vec.begin();
        y.it = rhs.m_vec.begin();
        std::vector<word_t>::iterator ir = res.m_vec.begin();
        while (x.it < m_vec.end()) {    // go through all words in m_vec
            if (x.nWords == 0)
                x.decode();
            if (y.nWords == 0)
                y.decode();
            if (x.isFill != 0) {
                if (y.isFill != 0) {
                    if (y.fillBit == 0) {
                        x -= y.nWords;
                        res.copy_fill(ir, y);
                    }
                    else if (y.nWords < x.nWords) {
                        x.nWords -= y.nWords;
                        y.fillBit = x.fillBit;
                        res.copy_fill(ir, y);
                    }
                    else {
                        res.copy_runs(ir, x, y.nWords);
                        y.it += (y.nWords == 0);
                    }
                }
                else if (x.fillBit == 0) {
                    y -= x.nWords;
                    res.copy_fill(ir, x);
                }
                else {
                    res.copy_runs(ir, y, x.nWords);
                    x.it += (x.nWords == 0);
                }
            }
            else if (y.isFill != 0) {
                if (y.fillBit == 0) {
                    x -= y.nWords;
                    res.copy_fill(ir, y);
                }
                else {
                    res.copy_runs(ir, x, y.nWords);
                    y.it += (y.nWords == 0);
                }
            }
            else { // both words are not compressed
                *ir = *x.it & *y.it;
                x.nWords = 0;
                y.nWords = 0;
                ++x.it;
                ++y.it;
                ++ir;
            }
        } // while (x.it < m_vec.end())

        /*if (x.it != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_d2 expects to exhaust i0 "
                "but there are " << (m_vec.end() - x.it) << " word(s) left";
            throw "and_d2 internal error";
        }

        if (y.it != rhs.m_vec.end()) {
            word_t nb0 = do_cnt() + active.nbits;
            word_t nb1 = rhs.do_cnt() + rhs.active.nbits;
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_d2 between two bit vectors of "
                "sizes " << nb0 << ':' << bytes() << " and "
                << nb1 << ':' << rhs.bytes() << "expects to exhaust i1 "
                "but there are " << (rhs.m_vec.end() - y.it) << " word(s) left";
            throw "and_d2 internal error";
        }

        if (ir != res.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::and_d2 expects to exhaust ir "
                "but there are " << (res.m_vec.end() - ir) << " word(s) left";
            throw "and_d2 internal error";
        }*/
    }

    // work with the two active_words
    res.active.val = active.val & rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "operand 1 of AND" << *this << std::endl;
    lg() << "operand 2 of AND" << rhs << std::endl;
    lg() << "result of AND " << res;
#endif
} // ibis::bitvector8::and_d2

// both operands of the 'xor' operation are not compressed
void ibis::bitvector8::xor_c0(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure *this is not shared!
    nset = 0;
    std::vector<word_t>::iterator i0 = m_vec.begin();
    std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
    while (i0 != m_vec.end()) {    // go through all words in m_vec
        *i0 ^= *i1;
        i0++;
        i1++;
    } // while (i0 != m_vec.end())

    // the last thing -- work with the two active_words
    active.val ^= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result XOR " << *this << std::endl;
    /*
      if (m_vec.size()*(MAXBITS<<1) >= nbits)
      decompress();
    */
#endif
} // ibis::bitvector8::xor_c0

// xor operation where rhs is not compressed (not one word is a counter)
void ibis::bitvector8::xor_c1(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    std::vector<word_t>::const_iterator i0 = m_vec.begin();
    std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin(), i2;
    word_t s0;
    res.clear();
    res.m_vec.reserve(rhs.m_vec.size());
    while (i0 != m_vec.end()) {    // go through all words in m_vec
        if (*i0 > ALLONES) { // i0 is compressed
            s0 = ((*i0) & MAXCNT);
            i2 = i1 + s0;
            res.nbits += s0 * MAXBITS;
            if ((*i0) >= HEADER1) { // the result is the compliment of i1
                for (; i1 != i2; i1++)
                    res.m_vec.push_back((*i1) ^ ALLONES);
            }
            else { // the result is *i1
                for (; i1 != i2; i1++)
                    res.m_vec.push_back(*i1);
            }
        }
        else { // both words are not compressed
            res.active.val = *i0 ^ *i1;
            res.append_active();
            i1++;
        }
        i0++;
    } // while (i0 != m_vec.end())

    if (i1 != rhs.m_vec.end()) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::xor_c1 expects to exhaust i1 but "
            "there are " << (rhs.m_vec.end() - i1) << " word(s) left";
        throw "xor_c1 iternal error";
    }

    // the last thing -- work with the two active_words
    res.active.val = active.val ^ rhs.active.val;
    res.active.nbits = active.nbits;

#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result XOR " << res;
#endif
} // ibis::bitvector8::xor_c1

// bitwise xor (^) operation -- both operands may contain compressed words
void ibis::bitvector8::xor_c2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    int bi_idx_pos = 0;
    run8 x, y;
    res.clear();
    x.it = m_vec.begin();
    y.it = rhs.m_vec.begin();
    res.m_vec.reserve(rhs.m_vec.size() + this->m_vec.size() + 1);
    while (x.it < m_vec.end()) {    // go through all words in m_vec
        if (x.nWords == 0)
            x.decode();
        if (y.nWords == 0)
            y.decode();
        if (x.isFill != 0) { // x points to a fill
            // if both x and y point to a fill, use the longer fill
            if (y.isFill != 0 && y.nWords >= x.nWords) {
                if (y.fillBit == 0)
                    res.copy_runs(x, y.nWords);
                else
                    res.copy_runsn(x, y.nWords);
                y.it += (y.nWords == 0);
            }
            else if (x.fillBit == 0) {
                res.copy_runs(y, x.nWords, rhs.index, bi_idx_pos);
                // res.copy_runs(y, x.nWords);
                x.it += (x.nWords == 0);
            }
            else {
                res.copy_runsn(y, x.nWords);
                x.it += (x.nWords == 0);
            }
        }
        else if (y.isFill != 0) {    // y points to a fill
            if (y.fillBit == 0)
                res.copy_runs(x, y.nWords);
            else
                res.copy_runsn(x, y.nWords);
            y.it += (y.nWords == 0);
        }
        else { // both words are not compressed
            res.active.val = *x.it ^ *y.it;
            res.append_active();
            x.nWords = 0;
            y.nWords = 0;
            ++x.it;
            ++y.it;
        }
    } // while (x.it < m_vec.end())

    /*if (x.it != m_vec.end()) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::xor_c2 expects to exhaust i0 but "
            "there are " << (m_vec.end() - x.it) << " word(s) left";
        // TODO: fix
        // throw "xor_c2 interal error";
    }

    if (y.it != rhs.m_vec.end()) {
        LOGGER(ibis::gVerbose > 0)
            << "Warning -- bitvector::xor_c2 expects to exhaust i1 but "
            "there are " << (rhs.m_vec.end() - y.it) << " word(s) left";
        LOGGER(ibis::gVerbose > 4)
            << "Bitvector 1\n" << *this << "\n Bitvector 2\n"
            << rhs << "\nXOR result so far\n" << res;
        // TODO: fix
        // throw "xor_c2 internal error";
    }*/

    // the last thing -- work with the two active_words
    res.active.val = active.val ^ rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "result of XOR " << res << std::endl;
#endif
} // bitvector8& ibis::bitvector8::xor_c2

// assuming *this is uncompressed but rhs is compressed, this function
// performs the XOR operation and overwrites *this with the result
void ibis::bitvector8::xor_d1(const ibis::bitvector8& rhs) {
    // m_vec.nosharing(); // make sure *this is not shared!
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "operand 1 of XOR" << *this << std::endl;
        lg() << "operand 2 of XOR" << rhs;
    }
#endif
    if (rhs.m_vec.size() == 1) {
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) { // complement every bit
            for (std::vector<word_t>::iterator i = m_vec.begin(); i != m_vec.end();
                ++i) {
                if (*i > ALLONES)
                    *i ^= FILLBIT;
                else
                    *i ^= ALLONES;
            }
            if (nset > 0)
                nset = nbits - nset;
        }
        else if (*it <= ALLONES) {
            m_vec[0] = (*it ^ *(rhs.m_vec.begin()));
            nset = cnt_ones(m_vec[0]);
        }
    }
    else if (rhs.m_vec.size() > 1) {
        nset = 0;
        word_t s0;
        std::vector<word_t>::iterator i0 = m_vec.begin();
        std::vector<word_t>::const_iterator i1 = rhs.m_vec.begin();
        while (i1 != rhs.m_vec.end()) {    // go through all words in m_vec
            if (*i1 > ALLONES) { // i1 is compressed
                s0 = ((*i1) & MAXCNT);
                if ((*i1) >= HEADER1) { // the result is the complement of i0
                    std::vector<word_t>::const_iterator stp = i0 + s0;
                    while (i0 < stp) {
                        *i0 ^= ALLONES;
                        ++i0;
                    }
                }
                else { // the result is *i0
                    i0 += s0;
                }
            }
            else { // both words are not compressed
                *i0 ^= *i1;
                ++i0;
            }
            ++i1;
        } // while (i1 != rhs.m_vec.end())

        if (i0 != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::xor_d1 expects to exhaust i0 but "
                "there are " << (m_vec.end() - i0) << " word(s) left";
            throw "xor_d1 internal error";
        }
    }

    // the last thing -- work with the two active_words
    active.val ^= rhs.active.val;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    {
        ibis::util::logger lg(4);
        lg() << "result of XOR" << *this;
    }
#endif
} // ibis::bitvector8::xor_d1

// xor operation on two compressed bitvectors, generates uncompressed result
void ibis::bitvector8::xor_d2(const ibis::bitvector8& rhs,
    ibis::bitvector8& res) const {
    // establish a uncompressed bitvector with the right size
    res.nbits = ((nbits == 0 && m_vec.size() > 0) ?
        (rhs.nbits == 0 && rhs.m_vec.size() > 0) ?
        (m_vec.size() <= rhs.m_vec.size() ? do_cnt() : rhs.do_cnt())
        : rhs.nbits : nbits);
    res.m_vec.resize(res.nbits / MAXBITS);

    // fill res with the right numbers
    if (m_vec.size() == 1) { // only one word in *this
        std::vector<word_t>::const_iterator it = m_vec.begin();
        if (*it > HEADER1) { // complement of rhs
            rhs.copy_comp(res.m_vec);
            if (rhs.nset > 0)
                res.nset = nbits - rhs.nset;
        }
        else if (*it > ALLONES) { // copy rhs
            rhs.decompress(res.m_vec);
            res.nset = rhs.nset;
        }
        else {
            res.m_vec[0] = (*it ^ *(rhs.m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (rhs.m_vec.size() == 1) { // only one word in rhs
        std::vector<word_t>::const_iterator it = rhs.m_vec.begin();
        if (*it > HEADER1) { // complement of *this
            copy_comp(res.m_vec);
            if (nset > 0) res.nset = nbits - nset;
        }
        else if (*it > ALLONES) { // copy *this
            decompress(res.m_vec);
            res.nset = nset;
        }
        else {
            res.m_vec[0] = (*it ^ *(m_vec.begin()));
            res.nset = cnt_ones(res.m_vec[0]);
        }
    }
    else if (m_vec.size() > 1) { // more than one word in *this
        run8 x, y;
        res.nset = 0;
        x.it = m_vec.begin();
        y.it = rhs.m_vec.begin();
        std::vector<word_t>::iterator ir = res.m_vec.begin();
        while (x.it < m_vec.end()) {    // go through all words in m_vec
            if (x.nWords == 0)
                x.decode();
            if (y.nWords == 0)
                y.decode();
            if (x.isFill != 0) {
                if (y.isFill != 0 && y.nWords >= x.nWords) {
                    if (y.fillBit == 0) {
                        res.copy_runs(ir, x, y.nWords);
                    }
                    else {
                        res.copy_runsn(ir, x, y.nWords);
                    }
                    y.it += (y.nWords == 0);
                }
                else {
                    if (x.fillBit == 0) {
                        res.copy_runs(ir, y, x.nWords);
                    }
                    else {
                        res.copy_runsn(ir, y, x.nWords);
                    }
                    x.it += (x.nWords == 0);
                }
            }
            else if (y.isFill != 0) {
                if (y.fillBit == 0) {
                    res.copy_runs(ir, x, y.nWords);
                }
                else {
                    res.copy_runsn(ir, x, y.nWords);
                }
                y.it += (y.nWords == 0);
            }
            else { // both words are not compressed
                *ir = *x.it ^ *y.it;
                x.nWords = 0;
                y.nWords = 0;
                ++x.it;
                ++y.it;
                ++ir;
            }
        } // while (x.it < m_vec.end())

        if (x.it != m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::xor_d2 expects to exhaust i0 but "
                "there are " << (m_vec.end() - x.it) << " word(s) left";
            throw "xor_d2 internal error";
        }

        if (y.it != rhs.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::xor_d2 expects to exhaust i1 but "
                "there are " << (rhs.m_vec.end() - y.it) << " word(s) left";
            throw "xor_d2 internal error";
        }

        if (ir != res.m_vec.end()) {
            LOGGER(ibis::gVerbose > 0)
                << "Warning -- bitvector::xor_d2 expects to exhaust ir but "
                "there are " << (res.m_vec.end() - ir) << " word(s) left";
            throw "xor_d2 internal error";
        }
    }

    // work with the two active_words
    res.active.val = active.val ^ rhs.active.val;
    res.active.nbits = active.nbits;
#if DEBUG + 0 > 1 || _DEBUG + 0 > 1
    ibis::util::logger lg(4);
    lg() << "operand 1 of XOR" << *this << std::endl;
    lg() << "operand 2 of XOR" << rhs << std::endl;
    lg() << "result of XOR " << res;
#endif
} // ibis::bitvector8::xor_d2
