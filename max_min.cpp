
double * minmax(double v[])
{

    double min1=9999999,max1 = -9999,len=0;
    static double out[2] = {0,0};
    int i=0;
    for(int i=0; v[i]; i++)
        len++;

    for(i=0; i<len; i++)
    {

        if(v[i]>max1)
        {
            max1=v[i];
        }
        if(v[i]<min1)
        {
            min1=v[i];
        }
    }
    out[0] = min1;
    out[1] = max1;
    printf("inside func%lf %lf\n",out[0],out[1]);
    return out;
}
