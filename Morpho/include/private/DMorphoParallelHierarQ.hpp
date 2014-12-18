/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_MORPHO_PARALLEL_HIERARQ_HPP
#define _D_MORPHO_PARALLEL_HIERARQ_HPP

#include "DMorphoHierarQ.hpp"

namespace smil
{

template <class T, class TokenType=UINT>
class ParHierarQInc
{
        protected:
                size_t gray_level_nbr;
                map<T, size_t> begin;
                map<T, size_t> end;
                T* tokens;
                vector<bool> assigned;
                vector<omp_lock_t> locks;
                size_t size;
                T higher_level;

                bool initialized;
        public:
                ParHierarQInc () : initialized (false), size(0), gray_level_nbr(0)
                {
                }
                ~ParHierarQInc ()
                {
                        for (int i=0; i<locks.size(); ++i)
                                omp_destroy_lock (&(locks[i]));
                }
                void initialize (const Image<T> &img)
                {
                        if (initialized)
                                reset();

                        size_t GRAY_LEVEL_NBR = ImDtTypes<T>::cardinal ();
                        size_t *h = new size_t[GRAY_LEVEL_NBR];
                        histogram (img, h);
                        size_t last_offset = 0;
                        #pragma        omp parallel for 
                        for (size_t i=0; i<GRAY_LEVEL_NBR; ++i)
                                if (h[i] != 0){
                                        begin[i] = last_offset;
                                        last_offset += h[i];
                                        end[i] = begin[i];
                                        gray_level_nbr++;
                                }
                        tokens = new T[gray_level_nbr];
                        assigned.resize (img.getPixelCount(), false);
                        locks.resize (gray_level_nbr);
                        #pragma omp parallel for
                        for (int i=0; i<gray_level_nbr; ++i)
                                omp_init_lock(&(locks[i]));
                                
                        higher_level = GRAY_LEVEL_NBR ;
                }
                void initialize_and_fill (const Image<T> &img) {
                        if (initialized)
                                reset();

                        size_t GRAY_LEVEL_NBR = ImDtTypes<T>::cardinal ();
                        size_t *h = new size_t[GRAY_LEVEL_NBR];
                        histogram (img, h);

                        size_t last_offset=0;

                        for (size_t i=0; i<GRAY_LEVEL_NBR; ++i)
                                if (h[i] != 0){
                                        begin[i] = last_offset;
                                        last_offset += h[i];
                                        end[i] = begin[i];
                                        gray_level_nbr++;
                                }
                        tokens = new T[gray_level_nbr];
                        assigned.resize (gray_level_nbr, true);

                        typename ImDtTypes<T>::lineType pixelsBegin = img.getPixels();
                        typename ImDtTypes<T>::lineType pixels;
                        size_t offset;

                        size = img.getPixelCount();

                        printSelf ();

//                        locks.resize (size);
//                        #pragma omp parallel for
//                        for (int i=0; i<gray_level_nbr; ++i) 
//                                omp_init_lock(&(locks[i]));


                        #pragma omp parallel for firstprivate(pixelsBegin) private(pixels,offset)
                        for (size_t i=0; i<size; ++i) {
                                pixels = pixelsBegin+i;
                                #pragma atomic capture 
                                offset = end[*pixels]++;
                                tokens[offset] = i;
                                assigned[offset] = true;

                        }

                        higher_level = begin.begin()->first;
                }
                void reset ()
                {
                        if (!initialized)
                                return;
                        begin.clear();
                        end.clear();
                        delete[] tokens;
                        assigned.clear();
                        for (int i=0; i<locks.size(); ++i)
                                omp_destroy_lock (&(locks[i]));
                        locks.clear();
                }
                size_t getSize()
                {
                        return size;
                }
                bool isEmpty ()
                {
                        return size==0;
                }
                size_t getHigherLevel ()
                {
                        return higher_level;
                }
                void printSelf (bool print_token = false) 
                {
                        cout << "===================" << endl;
                        cout << "gray_level_nbr: " << gray_level_nbr << endl;
                        cout << "size: " << (int)size << endl;
                        cout << "higher_level: " << (int)higher_level << endl;
                        cout << "===================" << endl;
                        for (typename map<T,size_t>::iterator it=begin.begin(); it!=begin.end(); it++){
                                 cout << (int)it->first << " " << (int)it->second <<  " " << (int)end[it->first] << endl;
                                if (print_token) {
                                        for (int i=it->second; i<=end[it->first];i++) {
                                                cout << "assigned: " << assigned[i] << ", token: " << (int)tokens[i] << endl;
                                        }
                                }
                        }
                        cout << "===================" << endl;
                }
                virtual void push (T value, TokenType token)
                {
                        size_t offset;
                        #pragma omp critical
                        {
                        if (value < higher_level)
                                higher_level = value;
                        offset = end[value]++;
                        }
                        tokens[offset] = token;
                        assigned[offset]=true;
                }
                virtual void findNewReferenceLevel()
                {
                        for (typename map<T,size_t>::iterator it=begin.begin(); it!=begin.end(); it++){
                                if (it->second != end[it->first]-1) {
                                        higher_level = it->first;
                                        break;
                                }
                        }
                }
                virtual bool par_pop (UINT id_thread, UINT nbr_threads, TokenType& result)
                {
                        size_t offset;
                        
                        return true;
                }
                virtual TokenType pop ()
                {
                        TokenType offset = tokens[begin[higher_level]];
                        assigned[begin[higher_level]]=false;
                        if (begin[higher_level] != end[higher_level]-1) {
                                begin[higher_level]++;
                        } else if (size > 1) {
                                findNewReferenceLevel();
                        }
                        size--;
                        return offset;
                }        
};

template <class T, class TokenType=UINT>
class ParHierarQDec : public ParHierarQInc<T,TokenType>
{
        public:
                ParHierarQDec () : ParHierarQInc<T,TokenType> () {}
                void push (T value, TokenType token)
                {
                        size_t offset;
                        #pragma omp critical
                        {
                        if (value < this->higher_level)
                                this->higher_level = value;
                        offset = this->end[value]++;
                        omp_unset_lock (&this->lock_higher_value);
                        }
                        this->tokens[offset] = token;
                        this->assigned[offset] = false;
                }
                void findNewReferenceLevel ()
                {
                        for (typename map<T,size_t>::iterator it=this->begin.rbegin(); it!=this->begin.rend(); it++){
                                if (it->second != this->end[it->first]) {
                                        this->higher_level = it->first;
                                        break;
                                }
                        }
                
                }
};

} // namespace smil

#endif // _D_MORPHO_PARALLEL_HIERARQ_HPP

