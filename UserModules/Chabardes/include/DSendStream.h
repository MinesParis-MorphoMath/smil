#ifndef _DSENDSTREAM_H_
#define _DSENDSTREAM_H_

namespace smil {
    template <class T>
    class SendStream {
        public:
            virtual RES_T next (Chunk<T> &c) {
                cout << "Don't know how to read." << endl;
            }
            virtual RES_T nexts (Chunk<T> *c, const int len_to_r) {
                cout << "Don't know how to read." << endl;
            }
            bool eof () {
                cout << "Don't know when it's ending." << endl;
                return true;
            }
    };

    template <class T>
    class SendArrayStream : public SendStream<T> {
        public:
            SendArrayStream () : r_head (0) {}
            bool eof () {
                return r_head >= nbr_chunks;
            }

            int nbr_chunks;
            int chunk_len;
            int r_head;
            int size[3];
            T* data;
    };

    template <class T>
    class SendArrayStream_chunk : public SendArrayStream<T> {
        public:
            const RES_T read_chunk (const int num, Chunk<T> &c) const {
                int x,y,z,s_x,s_y,s_z,w_x,w_y,w_z,w_s_x,w_s_y,w_s_z;
                w_x = num/(chunks_per_dim[0]*chunks_per_dim[1]);
                w_y = num/chunks_per_dim[0]-w_z*chunks_per_dim[1];
                w_z = num-w_y*chunks_per_dim[0]-w_z*chunks_per_dim[1]*chunks_per_dim[0];
                w_s_x = (w_x==chunks_per_dim[0]-1) ? this->size[0] - w_x*this->size[0]/chunks_per_dim[0] : this->size[0]/chunks_per_dim[0] ;
                w_x *= this->size[0]/chunks_per_dim[0];
                w_s_y = (w_x==chunks_per_dim[1]-1) ? this->size[1] - w_x*this->size[1]/chunks_per_dim[1] : this->size[1]/chunks_per_dim[1];
                w_y *= this->size[1]/chunks_per_dim[1];
                w_s_z = (w_x==chunks_per_dim[2]-1) ? this->size[2] - w_x*this->size[2]/chunks_per_dim[2] : this->size[2]/chunks_per_dim[2];
                w_z *= this->size[2]/chunks_per_dim[2];
                x = (w_x > intersect_width) ? w_x-intersect_width : 0;
                y = (w_y > intersect_width) ? w_y-intersect_width : 0;
                z = (w_z > intersect_width) ? w_z-intersect_width : 0;
                s_x = (s_x+w_s_x+intersect_width < this->size[0]) ? (w_x-x)+w_s_x+intersect_width : this->size[0]-x;
                s_y = (s_y+w_s_y+intersect_width < this->size[1]) ? (w_y-y)+w_s_y+intersect_width : this->size[1]-y;
                s_z = (s_z+w_s_z+intersect_width < this->size[2]) ? (w_z-z)+w_s_z+intersect_width : this->size[2]-z;

                c.createFromArray (
                            this->data,
                            this->size[0],
                            this->size[1],
                            x, y, z,
                            s_x, s_y, s_z,
                            w_x, w_y, w_z,
                            w_s_x, w_s_y, w_s_z
                        );
                return RES_OK;
            }
            const RES_T next (Chunk<T> &c) const {
                ASSERT (c.isInitialized() && this->r_head < this->nbr_chunks);
                return read_chunk (this->r_head++, c) ;
            }
            int intersect_width;
            int chunks_per_dim[3];
    };

    template <class T>
    class SendArrayStream_slice : public SendArrayStream<T> {
        public:
    };
}

#endif
