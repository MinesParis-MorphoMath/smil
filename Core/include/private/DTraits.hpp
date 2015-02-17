/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


#ifndef _D_TRAITS_HPP
#define _D_TRAITS_HPP



namespace smil
{
    template<bool C, typename T = void>
    struct enable_if 
    {
        typedef T type;
    };

    template<typename T>
    struct enable_if<false, T> { };

    template<typename, typename>
    struct is_same 
    {
        static bool const value = false;
    };

    template<typename A>
    struct is_same<A, A> 
    {
        static bool const value = true;
    };

    template<typename B, typename D>
    struct is_base_of
    {
      private:
          static D* m_d;
    
      private:
          static char check( B* );
          static long check( ... );
    
      public:
          static bool const value = sizeof check(m_d) == 1 &&  !is_same<B volatile const, void volatile const>::value;
    };
    
    #define ENABLE_IF(COND, RET_TYPE) typename smil::enable_if< ( COND ), RET_TYPE >::type
    #define IS_SAME(A, B) ( smil::is_same<A, B>::value )
    #define IS_DERIVED_FROM(D, B) ( smil::is_base_of<B, D>::value )
      
} // namespace smil

#endif // _D_TRAITS_HPP
