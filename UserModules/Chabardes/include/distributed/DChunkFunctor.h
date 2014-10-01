#ifndef _D_CHUNKFUNCTOR_H_
#define _D_CHUNKFUNCTOR_H_

#include <DGlobalHeader.h>
#include <DChunk.h>

namespace smil {

    template <class T>
    class chunkFunctor {
        public:
            auto_ptr<chunkFunctor<T> > clone () {return doClone();}
            virtual void operator() (Chunk<T> &c, const MPI_Comm &comm, const int rank, const GlobalHeader& gh) =0;
        private:
            virtual chunkFunctor<T> doClone() {return new (*this);}
    };

    template <class T>
    class chunkGradient : public chunkFunctor<T> {
        public:
            std::auto_ptr<chunkGradient<T> > clone () {return doClone();}
            void operator() (Chunk<T> &c, const MPI_Comm &comm, const int rank, const GlobalHeader& gh) {
                SharedImage<T> fakeIm (c.getData(), c.getSize(0), c.getSize(1), c.getSize(2));
                Image<T> tmp = Image<T> (fakeIm);

                erode (fakeIm, tmp, Cross3DSE());
                dilate (fakeIm, fakeIm, Cross3DSE());
                fakeIm -= tmp;
            }
        private:
            chunkGradient* doClone() {return new (*this);}
    };

    template <class T>
    class chunk_ref {
        public:
            chunk_ref (const T &d) {
                kill = true;
            }
            chunk_ref (T *d) {
                kill = false;
            }
            chunk_ref (const chunk_ref <T> &r) {
                kill = true;
                t = r.t?r.t->clone():NULL;
            }
            ~chunk_ref () {if (t & kill) delete t;}

            T* operator->() const {return t;}
            int operator< (const chunk_ref <T> &r) const {
                return t?r.t?(*t) < (*r.t) : false : true;
            }
            operator T&() const {return *t;}
            operator T*() const {return t;}
            T& operator*() const {return *t;}
        protected:
            T* t;
        private:
            bool kill;
    };

}

#endif
