def edit_distance(s1, s2):
    m = len(s1) + 1;
    n = len(s2) + 1;
    cost = 0;
    mTable = {};
    for i in range(0, m):
        for j in range(0, n):
            mTable[i,j] = 0;
    for i in range(0, m):
        mTable[i, 0] = i;
    for j in range(0, n):
        mTable[0, j] = j;
    for i in range(1, m):
        for j in range(1, n):
            if s1[i-1] == s2[j-1]:
                cost = 0;
            else:
                cost = 1;
            m1 = mTable[i-1, j] + 1;
            m2 = mTable[i, j-1] + 1;
            m3 = mTable[i-1, j-1] + cost;
            mTable[i, j] = min(m1, m2, m3);

    print("Edit Distance Matrix\n");
    print("       ", end = '');
    for j in range(0, n-1):
        print("| " + s2[j] + " ", end = '');
    print("\n");
    for i in range(0, m):
        if i == 0:
            print("   ", end='');
        if i>0:
            print(" " + s1[i-1] + " ", end='');
        for j in range(0, n):
            print("| " + str(mTable[i,j]) + " ", end='')
        print("\n");
    return mTable, mTable[m-1, n-1];

def get_edits(s1, s2, mTable, nEditDist): # Function to get the letters that need to be either inserted(or removed) or replaced to convert the smaller string into the larger string. 
    m = len(s1) + 1;
    n = len(s2) + 1;
    
    i_old = m-1;
    j_old = n-1;
    i_new = m-1;
    j_new = n-1;

    sOperation = "";
    nIndexOfOperation = nEditDist-1;
    sOperationList = {};

    for i in range(0, nEditDist-1):
        sOperationList[i] = "";
    while 1:
        nLeft = mTable[i_old, j_old-1];
        nUp = mTable[i_old-1, j_old];
        nUpLeft = mTable[i_old-1, j_old-1];
        if nUpLeft <= nLeft and nUpLeft <= nUp:
            i_new = i_old-1;
            j_new = j_old-1;
            if mTable[i_old, j_old] > nUpLeft:
                sOperation = (s2[j_old-1] if m > n else s1[i_old-1]) if i_old == j_old else (s1[i_old-1] if m > n else s2[j_old-1]);
                sOperationList[nIndexOfOperation] = sOperation;
                nIndexOfOperation -= 1;
        elif nLeft <= nUpLeft and nLeft <= nUp:
            i_new = i_old;
            j_new = j_old-1;
            if mTable[i_old, j_old] > nLeft:
                sOperation = (s2[j_old-1] if m > n else s1[i_old-1]) if i_old == j_old else (s1[i_old-1] if m > n else s2[j_old-1]);
                sOperationList[nIndexOfOperation] = sOperation;
                nIndexOfOperation -= 1;
        elif nUp <= nUpLeft and nUp <= nLeft:
            i_new = i_old-1;
            j_new = j_old;
            if mTable[i_old, j_old] > nUp:
                sOperation = (s2[j_old-1] if m > n else s1[i_old-1]) if i_old == j_old else (s1[i_old-1] if m > n else s2[j_old-1]);
                sOperationList[nIndexOfOperation] = sOperation;
                nIndexOfOperation -= 1;
        i_old = i_new;
        j_old = j_new;
        if i_old == 0 and j_old == 0:
            break;
    
    print("The sequence of edits:");
    for i in range(0, nEditDist):
        print("Step " + str(i+1) + " : " + sOperationList[i]);
    

if __name__=="__main__":
    #Example 1
    #sString1="kitten";
    #sString2="sitting";
    #Example 2
    #sString1 = "GAMBOL";
    #sString2 = "GUMBO";
    #Example 3
    sString1="writers";
    sString2="vintner";
    mTable, nEditDist = edit_distance(sString1, sString2);
    print("Edit distance is " + str(nEditDist));
    get_edits(sString1, sString2, mTable, nEditDist);
