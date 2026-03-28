import weka.filters.Filter;
import java.io.*;

/**
 * Klase laguntzailea Weka filtroak serializatu (gorde) eta
 * deserializatzeko (kargatzeko).
 *
 * 
 * Machine learning pipeline batean, entrenamenduan erabilitako
 * filtroak (adibidez: bektorizazioa edo atributu-aukeraketa)
 * gordetzea beharrezkoa da, inferentzian datu berriei transformazio
 * berdinak aplikatzeko.
 * 
 *
 * 
 * Klase honek {@link weka.filters.Filter} objektuak fitxategi
 * batean gordetzeko eta gero berriro kargatzeko aukera ematen du.
 * 
 *
 * 
 * Serializazioa Java-ko {@code ObjectOutputStream} eta
 * {@code ObjectInputStream} erabiliz egiten da.
 * 
 *
 * @version 1.0
 */
public class FilterSerializationHelper {

    /**
     * Weka filtro bat fitxategi batean gordetzen du (serializazioa).
     *
     * <p>
     * Metodo honek filtroaren egoera osoa gordetzen du, geroago
     * berrerabili ahal izateko inferentzia fasean.
     * </p>
     *
     * @param filter   Gorde nahi den filtroa
     * @param filepath Irteerako fitxategiaren bidea
     * @throws IOException Idaztean errorea gertatzen bada
     */
    public static void saveFilter(Filter filter, String filepath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filepath))) {
            oos.writeObject(filter);
        }
    }

    /**
     * Fitxategi batetik Weka filtro bat kargatzen du (deserializazioa).
     *
     * <p>
     * Kargatutako filtroa entrenamenduan erabilitako bera da,
     * eta datu berriei transformazio berdinak aplikatzeko erabiltzen da.
     * </p>
     *
     * @param filepath Kargatu nahi den fitxategiaren bidea
     * @return Kargatutako filtroa
     * @throws IOException            Irakurtzean errorea gertatzen bada
     * @throws ClassNotFoundException Klasea ezin bada aurkitu
     */
    public static Filter loadFilter(String filepath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filepath))) {
            return (Filter) ois.readObject();
        }
    }
}
