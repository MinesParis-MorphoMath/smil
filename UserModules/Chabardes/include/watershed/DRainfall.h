#ifndef _DRAINFALL_H_
#define _DRAINFALL_H_

template <class T, class labelT, class arrowT, class tokenT=UINT, class stackT=STD_Queue<tokenT> >
class waterdropFunc {
	public ;
		waterdropFunc () {}

		RES_T exec (cont Image<T> &in, Image<labelT> &label, Image<arrowT> &arrow, tokenT &loc, const StrElt &se=DEFAULT_SE) {

			

		}

		inline RES_T operator (cont Image<T> &in, Image<labelT> &label, Image<arrowT> &arrow, const StrElt &se=DEFAULT_SE) { return exec (in, ) ;}
	private :
		stackT stack;
};

RES_T arrowLowest () {
}

RES_T arrowGreatest () {

}

#endif // _DRAINFALL_H_
