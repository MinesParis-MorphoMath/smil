#ifndef _DRECVSTREAM_H_
#define _DRECVSTREAM_H_

namespace smil {
    template <class T>
    class RecvStream {
        public:
            RecvStream (Image<T> &im) {
                size_t s[3];
                im.getSize (s);
                for (int i=0; i<3; ++i) size[i] = s[i];
                data = im.getPixels (); 
            }
            RES_T write (Chunk<T> &c) {
                c.storeToArray (size[0], size[1], data);
            }
        private:
            unsigned int size[3];
            T* data;
    };
}

#endif
