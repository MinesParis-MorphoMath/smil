
%include smilCommon.i

SMIL_MODULE(smilZhangSkel)

%{
#include "DZhangSkel.hpp"
%}

%import smilCore.i
// %import smilBase.i

%include "DZhangSkel.hpp"

TEMPLATE_WRAP_FUNC(zhangSkeleton);
TEMPLATE_WRAP_FUNC(zhangThinning);
TEMPLATE_WRAP_FUNC(zhangSkeletonVarious);
