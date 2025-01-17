#include "deconv_hyper_lap.h"

const float BOUND[] = {0.82, 0.4610, 0.2590, 0.1450, 0.0820};
const float LEFT_LINE_P0[] = {1.08715491833576, 1.04404277230246, 1.02198575258148, 1.01061794069958, 1.00489456979082};
const float LEFT_LINE_P1[] = {0.161872978061302, 0.0670518176595489, 0.0284143580342271, 0.0120693414510977, 0.00505846638184741};
const float RIGHT_LINE_P0[] = {1.08715491833576, 1.04404277230246, 1.02198575258148, 1.01061794069958, 1.00489456979082};
const float RIGHT_LINE_P1[] = {-0.161872978061305, -0.0670518176595504, -0.0284143580342275, -0.0120693414510978, -0.00505846638184776};

void circular_conv(const cv::Mat& src, cv::Mat& dst, const cv::Mat& kernel, cv::Point anchor)
{
    cv::Mat tmp,out_tmp;
    int kernel_col = kernel.cols;
    int kernel_row = kernel.rows;
    copyMakeBorder(src,tmp,anchor.y,kernel_row - 1 - anchor.y, anchor.x,kernel_col - 1 - anchor.x,cv::BORDER_WRAP);
    filter2D(tmp,out_tmp,tmp.depth(),kernel,anchor);
    dst = cv::Mat(out_tmp,cv::Rect(anchor.x,anchor.y,src.cols,src.rows));
}

void divispectrum(const cv::Mat& X1, const cv::Mat& X2, cv::Mat& X_out)
{
    //ÕâÀïŒÙ¶šX2ÊÇÊµŸØÕó
    cv::Mat X_split[2];
    split(X1,X_split);
    divide(X_split[0],X2,X_split[0]);
    divide(X_split[1],X2,X_split[1]);

    merge(X_split,2,X_out);
}

void Solve_w(const cv::Mat& v, cv::Mat& w, int beta_chose)
{
    float bound = BOUND[beta_chose];
    float left_line_p0 = LEFT_LINE_P0[beta_chose];
    float left_line_p1 = LEFT_LINE_P1[beta_chose];
    float right_line_p0 = RIGHT_LINE_P0[beta_chose];
    float right_line_p1 = RIGHT_LINE_P1[beta_chose];
    float tmp;

    int mat_col = v.cols;
    int mat_row = v.rows;
    w.create(v.size(),v.type());
    if(v.isContinuous() && w.isContinuous())
    {
        mat_col = mat_col*mat_row;
        mat_row = 1;

    }

    for(int i = 0;i<mat_row;i++)
    {
        const float* vptr = v.ptr<float>(i);
        float * wptr = w.ptr<float>(i);
        for(int j = 0; j<mat_col;j++)
        {
            tmp = vptr[j];
            if(tmp<-bound)
            {
                wptr[j] = tmp * left_line_p0 + left_line_p1;
            }
            else if(tmp<bound)
            {
                wptr[j] = 0;
            }
            else
            {
                wptr[j] = tmp * right_line_p0 + right_line_p1;
            }
        }
    }
}

void dft_fftw(const cv::Mat& src, cv::Mat& dst)
{
    int mat_row = src.rows;
    int mat_col = src.cols;

    float * data_tmp = (float *)fftwf_malloc(sizeof(float)*mat_row*mat_col);
    fftwf_complex * out_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*mat_row*(mat_col/2+1));
    fftwf_plan p = fftwf_plan_dft_r2c_2d(mat_row,mat_col,data_tmp,out_tmp, FFTW_ESTIMATE);

    int row = mat_row;
    int col = mat_col;

    if(src.isContinuous())
    {
        col = mat_col * mat_row;
        row = 1;
    }
    for(int i = 0; i<row;i++)
    {
        const float* data_ptr = src.ptr<float>(i);
        memcpy((void *)(data_tmp + col*i),(void*)data_ptr ,sizeof(float)*col);
    }

    fftwf_execute(p);

    cv::Mat out_mat[2];
    out_mat[0].create(mat_row,mat_col/2+1,CV_32F);out_mat[1].create(mat_row,mat_col/2+1,CV_32F);

    col = out_mat[0].cols; row = out_mat[0].rows;
    if(out_mat[0].isContinuous())
    {
        col = row*col;
        row = 1;
    }
    for(int i = 0; i<row ;i++)
    {
        float* data_ptr = out_mat[0].ptr<float>(i);
        for(int j = 0;j<col;j++)
        {
            data_ptr[j] = out_tmp[i*col+j][0];
        }
    }

    col = out_mat[1].cols; row = out_mat[1].rows;
    if(out_mat[1].isContinuous())
    {
        col = row*col;
        row = 1;
    }
    for(int i = 0; i<row ;i++)
    {
        float* data_ptr = out_mat[1].ptr<float>(i);
        for(int j = 0;j<col;j++)
        {
            data_ptr[j] = out_tmp[i*col+j][1];
        }
    }
    merge(out_mat,2,dst);

    fftwf_free(data_tmp);
    fftwf_free(out_tmp);
    fftwf_destroy_plan(p);
}

void idft_fftw(const cv::Mat& src, cv::Mat& dst)
{
    int dst_col = src.cols*2 - 1;
    int dst_row = src.rows;
    cv::Mat src_tmp[2];

    fftwf_complex * data_tmp = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*src.rows*src.cols);
    float * out_tmp = (float *)fftwf_malloc(sizeof(float)*dst_col*dst_row);
    fftwf_plan p = fftwf_plan_dft_c2r_2d(dst_row,dst_col,data_tmp,out_tmp,FFTW_ESTIMATE);
    int col = src.cols;
    int row = src.rows;

    split(src,src_tmp);
    if(src_tmp[0].isContinuous())
    {
        col = col*row;
        row = 1;
    }
    for(int i = 0; i<row; i++)               //°ÑMatÀïµÄÊýŸÝ×°ÈëfftwÖž¶šµÄÄÚŽæ
    {
        float *real_ptr = src_tmp[0].ptr<float>(i);
        for(int j = 0; j<col; j++)
        {
            data_tmp[i*col+j][0] = real_ptr[j];
        }
    }
    if(src_tmp[1].isContinuous())
    {
        col = col*row;
        row = 1;
    }
    for(int i = 0; i<row; i++)               //°ÑMatÀïµÄÊýŸÝ×°ÈëfftwÖž¶šµÄÄÚŽæ
    {
        float *im_ptr = src_tmp[1].ptr<float>(i);
        for(int j = 0; j<col; j++)
        {
            data_tmp[i*col+j][1] = im_ptr[j];
        }
    }

    fftwf_execute(p);

    dst.create(dst_row,dst_col,CV_32F);
    row = dst_row;
    col = dst_col;

    if(dst.isContinuous())
    {
        col = row*col;
        row = 1;
    }
    for(int i = 0;i<row;i++)  //Ö»¶ÔÊµ²¿žÐÐËÈ€
    {
        float * ptr = dst.ptr<float>(i);
        for(int j =0 ; j<col; j++)
        {
            ptr[j] = out_tmp[i*col+j]/dst_col/dst_row;
        }
    }

    fftwf_free(data_tmp);
    fftwf_free(out_tmp);
    fftwf_destroy_plan(p);
}

void fast_deblurring(const cv::Mat& src_im, const cv::Mat& kernel,cv::Mat& yout)
{
    cv::Mat tmp;
    float beta[Iteration_time] = { 2.828427,8.000000,22.627417,64.000000,181.019336 };  //betaÊÇÇóœâ·œ³ÌµÄÏ¡Êè±äÁ¿£¬Ã¿ŽÎµüŽúÔöŽó
    float lambda = 5e2;
    float lambda_step = 2.828427124746190 ;//2*sqrt(2)

    cv::Mat dx = (cv::Mat_<float>(1,2) << 1,-1);
    cv::Mat dy = (cv::Mat_<float>(2,1) << 1,-1);
    cv::Mat dx_flip = (cv::Mat_<float>(1,2) << -1,1);
    cv::Mat dy_flip = (cv::Mat_<float>(2,1) << -1,1);

    cv::Mat Denorm1,Denorm2,ky,dx_extended,dy_extended,k_extended;

    //ŒÆËãÇóœâ¹«ÊœÖÐÒ»Ð©³£ÊýÏî
    //    ky  -- F(K)'*F(y)
    //    Denorm2  -- |F(K)|.^2
    //    Denorm1  -- |F(D^1)|.^2 + |F(D^2)|.^2

    copyMakeBorder(dx,dx_extended,0,src_im.rows - dx.rows,0, src_im.cols - dx.cols, cv::BORDER_CONSTANT,0);
    copyMakeBorder(dy,dy_extended,0,src_im.rows - dy.rows,0, src_im.cols - dy.cols, cv::BORDER_CONSTANT,0);
    copyMakeBorder(kernel,k_extended,0,src_im.rows - kernel.rows,0, src_im.cols - kernel.cols, cv::BORDER_CONSTANT,0);

    circular_conv(src_im,tmp,kernel, cv::Point(kernel.cols/2,kernel.rows/2));
    circular_conv(dx_extended,dx_extended,dx, cv::Point(0,0)); //ÕâÀïdx_extendedºÍdy_extended×öÖÐŒä±äÁ¿
    circular_conv(dy_extended,dy_extended,dy, cv::Point(0,0));
    //dft(tmp,ky,DFT_COMPLEX_OUTPUT);
    dft_fftw(tmp,ky);

    Denorm1 = dx_extended + dy_extended;   //ŽËŽŠDenorm1 = dx * dx + dy * dy (*ÊÇcorrelateÔËËã)
    //dft(Denorm1,Denorm1,DFT_COMPLEX_OUTPUT);
    //dft(k_extended,Denorm2,DFT_COMPLEX_OUTPUT);
    dft_fftw(Denorm1,Denorm1);
    dft_fftw(k_extended,Denorm2);

    cv::Mat split_tmp[2];
    split(Denorm1,split_tmp);
    split_tmp[0].copyTo(Denorm1);
    split(Denorm2,split_tmp);
    magnitude(split_tmp[0],split_tmp[1],Denorm2);
    pow(Denorm2,2,Denorm2);

    yout = src_im; //yout ÊÇµüŽúÖÐµÄœâ£¬³õÊŒ»¯Îª¶ÁÈëµÄÍŒÏñ,WÊÇµüŽú¹ý³ÌÖÐµÄÒ»žöÖÐŒäÁ¿
    cv::Mat youtx, youty;
    cv::Mat Wx,Wy,Wxx,Wyy; //Wx Wy·Ö±ðÊÇWµÄË®Æœ£¬Ž¹Ö±²î·Ö£¬Wxx£¬WyyÔòÊÇ¶ÔÓŠµÄ¶þœ×²î·Ö
    cv::Mat Denom;
    for(int i = 3; i<Iteration_time; i++)
    {
        Denom = beta[i]/lambda*(Denorm1) +Denorm2;  //K_2 ŽËŽŠÎª |F(D^1)|.^2 + |F(D^2)|.^2+  beta/lambda * |F(K)|.^2

        circular_conv(yout,youtx,dx_flip,cv::Point(1,0)); //Ë®Æœ²î·Ö
        circular_conv(yout,youty,dy_flip,cv::Point(0,1)); //ÊúÖ±²î·Ö

        Solve_w(youtx,Wx,i);
        Solve_w(youty,Wy,i);

        circular_conv(Wx,Wxx,dx,cv::Point(0,0));
        circular_conv(Wy,Wyy,dy,cv::Point(0,0));

        Wxx += Wyy;

        //dft(Wxx,Wxx,DFT_COMPLEX_OUTPUT); //ŽËŽŠWxx ÊÇWË®Æœ£¬Ž¹Ö±¶þœ×²î·ÖºÍµÄžµÀïÒ¶±ä»»
        dft_fftw(Wxx,Wxx);

        yout = ( ky + beta[i]/lambda* Wxx);
        //cout<<yout<<endl<<endl;

        cv::Mat tmp2;
        divispectrum(yout,Denom,tmp2);

        //dft(tmp2,yout,DFT_INVERSE|DFT_SCALE);
        idft_fftw(tmp2,yout);

        lambda *= lambda_step;
    }
}
