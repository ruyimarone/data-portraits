import re


consecutive_newlines = re.compile(r'[\r\n]+')
consecutive_spaces = re.compile(r'[^\S\n]+')
leading_spaces = re.compile(r'\n\s')

def proc_code(text):
    # replace consecutive carriage returns or newlines with a single newline
    text = consecutive_newlines.sub('\n', text)

    # replace consecutive whitespace with a single space
    text = consecutive_spaces.sub(' ', text)

    # whitespace has been squashed already, but now strip leading whitespace from each line
    text = leading_spaces.sub('\n', text)

    return text.strip()

if __name__ == '__main__':
    simple_python = """
    for i in range(100):
        if i:
            print('hello')
        else

            pass


    print('done')
    """

    fast_inv_sqrt = """
    float math_rsqrt(float number)
    {
        long i;
        float x2, y;
        const float threehalfs = 1.5F;

        x2 = number * 0.5F;
        y  = number;
        i  = * ( long * ) &y;                       // evil floating point bit level hacking（对浮点数的邪恶位级hack）
        i  = 0x5f3759df - ( i >> 1 );               // what the fuck?（这他妈的是怎么回事？）
        y  = * ( float * ) &i;
        y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration （第一次牛顿迭代）
        y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed（第二次迭代，可以删除）

        return y;
    }
    """

    # has some mixed indentation, on purpose
    fib_py ="""

    def fibRec(n):
    		if n < 2:
                  return n
            #     filler comment    
            else:
            	return fibRec(n-1) + fibRec(n-2)


    """

    mixed_lines = "hello\nworld\r\nthis\n\nis\r\n\ra\rtest\n"

    for code in [mixed_lines, simple_python, fast_inv_sqrt, fib_py]:
        print(proc_code(code))
        print('-' * 60)
        print(repr(proc_code(code)))
        print('-' * 60)

    import timers
    N = 10000
    with timers.Timer(f"Loop {N} times") as t:
        for i in range(N):
            proc_code(fast_inv_sqrt)

    print("{:.4} seconds per call".format(t.elapsed / N))

