#ifndef _DSENDSTREAM_H_
#define _DSENDSTREAM_H_

namespace smil {
    template <class T>
    class SendStream {
        protected:
        public:
    };

    template <class T>
    class SendChunkStream : public SendStream<T> {
        private:
        public:
    };

    template <class T>
    class SendSliceStream : public SendStream<T> {
        private:
        public:
    };
}

#endif
